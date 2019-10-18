import os
import datetime as dt
import re
try:
    import cartopy
except ImportError:
    # ReadTheDocs unable to pip install cartopy
    pass
import glob
import json
import pandas as pd
import numpy as np
import netCDF4
try:
    import cf_units
except ImportError:
    # ReadTheDocs unable to pip install cf-units
    pass
from forest import (
        gridded_forecast,
        satellite,
        rdt,
        earth_networks,
        geo,
        disk)
import bokeh.models
from collections import OrderedDict, defaultdict
from functools import partial
import scipy.ndimage
try:
    import shapely.geometry
except ImportError:
    # ReadTheDocs unable to pip install shapely
    pass
from forest.util import (
        timeout_cache,
        initial_time,
        coarsify)
from forest.exceptions import SearchFail


# Application data shared across documents
LOADERS = {}
IMAGES = OrderedDict()
VECTORS = OrderedDict()
COASTLINES = {
    "xs": [],
    "ys": []
}
BORDERS = {
    "xs": [],
    "ys": []
}
LAKES = {
    "xs": [],
    "ys": []
}
DISPUTED = {
    "xs": [],
    "ys": []
}


def on_server_loaded():
    global DISPUTED
    global COASTLINES
    global LAKES
    global BORDERS
    # Load coastlines/borders
    EXTENT = (-10, 50, -20, 10)
    COASTLINES = load_coastlines()
    LAKES = xs_ys(iterlines(
        cartopy.feature.NaturalEarthFeature(
            'physical',
            'lakes',
            '10m').intersecting_geometries(EXTENT)))
    DISPUTED = xs_ys(iterlines(
            cartopy.feature.NaturalEarthFeature(
                "cultural",
                "admin_0_boundary_lines_disputed_areas",
                "50m").geometries()))
    BORDERS = xs_ys(iterlines(
        cartopy.feature.NaturalEarthFeature(
            'cultural',
            'admin_0_boundary_lines_land',
            '50m').geometries()))


def add_loader(name, loader):
    global LOADERS
    if name not in LOADERS:
        LOADERS[name] = loader


def load_coastlines():
    return xs_ys(iterlines(
            cartopy.feature.COASTLINE.geometries()))


def xs_ys(lines):
    xs, ys = [], []
    for lons, lats in lines:
        x, y = geo.web_mercator(lons, lats)
        xs.append(x)
        ys.append(y)
    return {
        "xs": xs,
        "ys": ys
    }


def iterlines(geometries):
    def xy(g):
        if isinstance(g, shapely.geometry.LineString):
            return g.xy
        else:
            return g.exterior.coords.xy
    for geometry in geometries:
        try:
            for g in geometry:
                yield xy(g)
        except TypeError:
            yield xy(geometry)


class FileLocator(object):
    """Base class for file system locators"""
    @timeout_cache(dt.timedelta(minutes=10))
    def find(self, pattern):
        return sorted(glob.glob(pattern))


class GPM(FileLocator):
    def __init__(self, pattern):
        self.pattern = pattern
        super().__init__()

    def image(self, itime):
        return load_image(
                self.paths[0],
                "precipitation_flux",
                0,
                itime)


def cache(name):
    store = globals()[name]
    def decorator(f):
        def wrapped(*args):
            if args not in store:
                store[args] = f(*args)
            return store[args]
        return wrapped
    return decorator


class ActiveViewer(object):
    def __init__(self):
        self.active = False
        self.loaded_state = None
        self.pending_state = None

    def add_figure(self, figure):
        raise Exception("this method should be implemented")

    def connect_data(self, figure, renderer):
        self.active = True
        if self.pending_state is not None:
            self.load(self.pending_state)
            self.loaded_state = self.pending_state
            self.pending_state = None
        def hide(renderer):
            renderer.visible = False
            self.active = False
        return partial(hide, renderer)

    def on_state(self, state):
        if self.active:
            if (self.loaded_state != state):
                self.load(state)
                self.loaded_state = state
            self.pending_state = None
        else:
            self.pending_state = state


class WindBarbs(ActiveViewer):
    def __init__(self, paths):
        self.paths = paths
        self.finder = Finder(paths)
        self.source = bokeh.models.ColumnDataSource({
            "x": [],
            "y": [],
            "u": [],
            "v": []})
        super().__init__()

    def add_figure(self, figure):
        renderer = figure.barb(
                x="x",
                y="y",
                u="u",
                v="v",
                source=self.source)
        return super().connect_data(figure, renderer)

    def load(self, state):
        time_step, ipressure = state
        if time_step is None:
            return
        if ipressure is None:
            return
        path = self.find_path(time_step.initial)
        # itime = self.time_index(time_step.length)
        itime = time_step.index
        self.source.data = self.load_data(
                path,
                itime,
                ipressure)

    def find_path(self, initial_time):
        return self.finder.find(initial_time)

    def time_index(self, length):
        return 0

    @staticmethod
    @cache("VECTORS")
    def load_data(path, itime, ipressure):
        step = 100
        with netCDF4.Dataset(path) as dataset:
            var = dataset.variables["x_wind"]
            if len(var.dimensions) == 4:
                values = var[itime, ipressure]
            else:
                values = var[itime]
            u = values[::step, ::step]
            var = dataset.variables["y_wind"]
            if len(var.dimensions) == 4:
                values = var[itime, ipressure]
            else:
                values = var[itime]
            v = values[::step, ::step]
            for d in var.dimensions:
                if "longitude" in d:
                    lons = dataset.variables[d][::step]
                if "latitude" in d:
                    lats = dataset.variables[d][::step]
        gx, _ = geo.web_mercator(
            lons,
            np.zeros(len(lons), dtype="d"))
        _, gy = geo.web_mercator(
            np.zeros(len(lats), dtype="d"),
            lats)
        x, y = np.meshgrid(gx, gy)
        u = convert_units(u, 'm s-1', 'knots')
        v = convert_units(v, 'm s-1', 'knots')
        return {
            "x": x.flatten(),
            "y": y.flatten(),
            "u": u.flatten(),
            "v": v.flatten()
        }


class DBLoader(object):
    def __init__(self, name, pattern, locator):
        self.name = name
        self.pattern = pattern
        self.locator = locator

    def image(self, state):
        if not self.valid(state):
            return gridded_forecast.empty_image()

        try:
            path, pts = self.locator.locate(
                self.pattern,
                state.variable,
                state.initial_time,
                state.valid_time,
                state.pressure)
        except SearchFail:
            return gridded_forecast.empty_image()

        units = self.read_units(path, state.variable)
        data = load_image_pts(
                path,
                state.variable,
                pts,
                pts)
        if (len(state.pressures) > 0) and (state.pressure is not None):
            level = "{} hPa".format(int(state.pressure))
        else:
            level = "Surface"
        data.update(gridded_forecast.coordinates(state.valid_time,
                                                 state.initial_time,
                                                 state.pressures,
                                                 state.pressure))
        data["name"] = [self.name]
        data["units"] = [units]
        return data

    @staticmethod
    def read_units(filename,parameter):
        dataset = netCDF4.Dataset(filename)
        veep = dataset.variables[parameter]
        # read the units and assign a blank value if there aren't any:
        units = getattr(veep, 'units', '')
        dataset.close()
        return units


    def valid(self, state):
        if state.variable is None:
            return False
        if state.initial_time is None:
            return False
        if state.valid_time is None:
            return False
        if state.pressures is None:
            return False
        if len(state.pressures) > 0:
            if state.pressure is None:
                return False
            if not self.has_pressure(state.pressures, state.pressure):
                return False
        return True

    def has_pressure(self, pressures, pressure, tolerance=0.01):
        if isinstance(pressures, list):
            pressures = np.array(pressures)
        return any(np.abs(pressures - pressure) < tolerance)


class SeriesLoader(object):
    def __init__(self, paths):
        self.locator = SeriesLocator(paths)

    @classmethod
    def from_pattern(cls, pattern):
        return cls(sorted(glob.glob(os.path.expanduser(pattern))))

    def series(self,
            initial_time,
            variable,
            lon0,
            lat0,
            pressure=None):
        data = {"x": [], "y": []}
        paths = self.locator.locate(initial_time)
        for path in paths:
            segment = self.series_file(
                    path,
                    variable,
                    lon0,
                    lat0,
                    pressure=pressure)
            data["x"] += list(segment["x"])
            data["y"] += list(segment["y"])
        return data

    def series_file(self, *args, **kwargs):
        try:
            times, values = self._load_netcdf4(*args, **kwargs)
        except:
            times, values = self._load_cube(*args, **kwargs)
        return {
            "x": times,
            "y": values}

    def _load_cube(self, path, variable, lon0, lat0, pressure=None):
        import iris
        cube = iris.load_cube(path, iris.Constraint(variable))
        coord = cube.coord('time')
        times = netCDF4.num2date(coord.points, units=str(coord.units))
        lons = cube.coord('longitude').points
        lats = cube.coord('latitude').points
        if lons.ndim == 2:
            lons = lons[0, :]
        if lats.ndim == 2:
            lats = lats[:, 0]
        lons = geo.to_180(lons)
        i = np.argmin(np.abs(lons - lon0))
        j = np.argmin(np.abs(lats - lat0))
        values = cube.data

        # Index array based on coordinate ordering
        pts = []
        for c in cube.coords():
            if c.name() == 'longitude':
                pts.append(i)
            elif c.name() == 'latitude':
                pts.append(j)
            else:
                pts.append(slice(None))
        pts = tuple(pts)
        values = values[pts]

        # Filter time/pressure axes to select correct pressure
        if self._has_dim(cube, 'pressure'):
            pressures = cube.coord('pressure').points
            if len(cube.ndim) == 3:
                pts = self.search(pressures, pressure)
                values = values[pts]
                if isinstance(times, dt.datetime):
                    times = [times]
                else:
                    times = times[pts]
            else:
                mask = self.search(pressures, pressure)
                values = values[:, mask][:, 0]
        return times, values

    @staticmethod
    def _has_dim(cube, label):
        import iris
        try:
            cube.coord(label)
        except iris.exceptions.CoordinateNotFoundError:
            return False
        return True

    def _load_netcdf4(self, path, variable, lon0, lat0, pressure=None):
        with netCDF4.Dataset(path) as dataset:
            try:
                var = dataset.variables[variable]
            except KeyError:
                return [], []
            lons = geo.to_180(self._longitudes(dataset, var))
            lats = self._latitudes(dataset, var)
            i = np.argmin(np.abs(lons - lon0))
            j = np.argmin(np.abs(lats - lat0))
            times = self._times(dataset, var)
            values = var[..., j, i]
            if (
                    ("pressure" in var.coordinates) or
                    ("pressure" in var.dimensions)):
                pressures = self._pressures(dataset, var)
                if len(var.dimensions) == 3:
                    pts = self.search(pressures, pressure)
                    values = values[pts]
                    if isinstance(times, dt.datetime):
                        times = [times]
                    else:
                        times = times[pts]
                else:
                    mask = self.search(pressures, pressure)
                    values = values[:, mask][:, 0]
        return times, values

    @staticmethod
    def _times(dataset, variable):
        """Find times related to variable in dataset"""
        time_dimension = variable.dimensions[0]
        coordinates = variable.coordinates.split()
        for c in coordinates:
            if c.startswith("time"):
                try:
                    var = dataset.variables[c]
                    return netCDF4.num2date(var[:], units=var.units)
                except KeyError:
                    pass
        for v, var in dataset.variables.items():
            if len(var.dimensions) != 1:
                continue
            if v.startswith("time"):
                d = var.dimensions[0]
                if d == time_dimension:
                    return netCDF4.num2date(var[:], units=var.units)

    def _pressures(self, dataset, variable):
        return self._dimension("pressure", dataset, variable)

    def _longitudes(self, dataset, variable):
        return self._dimension("longitude", dataset, variable)

    def _latitudes(self, dataset, variable):
        return self._dimension("latitude", dataset, variable)

    @staticmethod
    def _dimension(prefix, dataset, variable):
        for d in variable.dimensions:
            if not d.startswith(prefix):
                continue
            if d in dataset.variables:
                return dataset.variables[d][:]
        for c in variable.coordinates.split():
            if not c.startswith(prefix):
                continue
            if c in dataset.variables:
                return dataset.variables[c][:]

    @staticmethod
    def search(pressures, pressure, rtol=0.01):
        return np.abs(pressures - pressure) < (rtol * pressure)


class UMLoader(object):
    def __init__(self, paths, name="UM", finder=None):
        self.name = name
        self.path = None
        self.paths = paths
        if finder is None:
            finder = Finder(paths)
        self.finder = finder
        with netCDF4.Dataset(self.paths[0]) as dataset:
            self.dimensions = self.load_dimensions(dataset)
            self.dimension_variables = self.load_dimension_variables(dataset)
            self.times = self.load_times(dataset)
            self.variables = self.load_variables(dataset)

    @staticmethod
    def load_variables(dataset):
        variables = []
        for v in dataset.variables:
            if "bnds" in v:
                continue
            if v in dataset.dimensions:
                continue
            if len(dataset.variables[v].dimensions) < 2:
                continue
            variables.append(v)
        return variables

    @staticmethod
    def load_times(dataset):
        times = {}
        for v in dataset.variables:
            var = dataset.variables[v]
            if len(var.dimensions) != 1:
                continue
            if v.startswith("time"):
                d = var.dimensions[0]
                times[d] = netCDF4.num2date(
                        var[:],
                        units=var.units)
        return times

    @staticmethod
    def load_dimensions(dataset):
        return {v: var.dimensions
            for v, var in dataset.variables.items()}

    @staticmethod
    def load_dimension_variables(dataset):
        return {d: dataset.variables[d][:]
                for d in dataset.dimensions
                if d in dataset.variables}

    def image(self, variable, pressure, itime):
        try:
            dimension = self.dimensions[variable][0]
        except KeyError as e:
            if variable == "precipitation_flux":
                variable = "stratiform_rainfall_rate"
                dimension = self.dimensions[variable][0]
            else:
                raise e
        times = self.times[dimension]
        valid = times[itime]
        initial = times[0]
        hours = (valid - initial).total_seconds() / (60*60)
        length = "T{:+}".format(int(hours))
        if hasattr(self.finder, 'path_points'):
            self.path, pts = self.finder.path_points(
                    initial,
                    valid,
                    pressure)
            data = load_image_pts(
                    self.path,
                    variable,
                    pts,
                    pts)
        else:
            self.path, ipressure = self.finder.find(
                    initial,
                    pressure,
                    variable)
            data = load_image(
                    self.path,
                    variable,
                    ipressure,
                    itime)
        if variable in self.pressure_variables:
            level = "{} hPa".format(int(pressure))
        else:
            level = "Surface"
        data["name"] = [self.name]
        data["valid"] = [valid]
        data["initial"] = [initial]
        data["length"] = [length]
        data["level"] = [level]
        return data

    def longitudes(self, variable):
        return self._lookup("longitude", variable)

    def latitudes(self, variable):
        return self._lookup("latitude", variable)

    def _lookup(self, prefix, variable):
        dims = self.dimensions[variable]
        for dim in dims:
            if dim.startswith(prefix):
                return self.dimension_variables[dim]


class SeriesLocator(object):
    """Helper to find files related to Series"""
    def __init__(self, paths):
        self.paths = paths
        self.table = defaultdict(list)
        for path in paths:
            time = initial_time(path)
            if time is None:
                try:
                    with netCDF4.Dataset(path) as dataset:
                        var = dataset.variables["forecast_reference_time"]
                        time = netCDF4.num2date(var[:], units=var.units)
                except KeyError:
                    continue
            self.table[self.key(time)].append(path)

    def initial_times(self):
        return np.array(list(self.table.keys()),
                dtype='datetime64[s]')

    def locate(self, initial_time):
        if isinstance(initial_time, str):
            return self.table[initial_time]
        if isinstance(initial_time, np.datetime64):
            initial_time = initial_time.astype(dt.datetime)
        return self.table[self.key(initial_time)]

    __getitem__ = locate

    def key(self, time):
        return "{:%Y-%m-%d %H:%M:%S}".format(time)


class Finder(object):
    def __init__(self, paths):
        self.paths = paths
        self.table = {
                initial_time(p): p for p in paths}
        self.initial_times = np.array(
                [initial_time(p) for p in paths],
                dtype='datetime64[s]')
        with netCDF4.Dataset(self.paths[0]) as dataset:
            if "pressure" in dataset.variables:
                self.pressures = dataset.variables["pressure"][:]
            else:
                self.pressures = None
            self.pressure_variables = self.load_heights(dataset)

    @staticmethod
    def load_heights(dataset):
        variables = set()
        for variable, var in dataset.variables.items():
            if variable == "pressure":
                continue
            if "pressure" in var.dimensions:
                variables.add(variable)
        return variables

    def find(self, initial, pressure, variable):
        if variable in self.pressure_variables:
            ipressure = 0
        else:
            ipressure = self.pressure_index(
                    self.pressures,
                    pressure)
        return self.find_path(initial), ipressure

    def pressure_index(self, pressures, pressure):
        if isinstance(pressures, list):
            pressures = np.array(pressures, dtype="f")
        return np.argmin(np.abs(pressures - pressure))

    def find_path(self, initial_time):
        try:
            return self.table[initial_time]
        except KeyError:
            initial_time = np.datetime64(
                    initial_time, 's')
            i = np.argmin(
                    np.abs(
                        self.initial_times - initial_time))
            return self.paths[i]


def pts_hash(pts):
    if isinstance(pts, np.ndarray):
        return pts.tostring()
    else:
        return pts


def load_image(path, variable, itime, ipressure):
    return load_image_pts(path, variable, (itime,), (itime, ipressure))


def load_image_pts(path, variable, pts_3d, pts_4d):
    key = (path, variable, pts_hash(pts_3d), pts_hash(pts_4d))
    if key in IMAGES:
        return IMAGES[key]
    else:
        try:
            lons, lats, values, units = _load_netcdf4(path, variable, pts_3d, pts_4d)
        except:
            lons, lats, values, units = _load_cube(path, variable, pts_3d, pts_4d)

    # Units
    if variable in ["precipitation_flux", "stratiform_rainfall_rate"]:
        if units == "mm h-1":
            values = values
        else:
            values = convert_units(values, units, "kg m-2 hour-1")
    elif units == "K":
        values = convert_units(values, "K", "Celsius")

    # Coarsify images
    threshold = 200 * 200  # Chosen since TMA WRF is 199 x 199
    if values.size > threshold:
        fraction = 0.25
    else:
        fraction = 1.
    lons, lats, values = coarsify(
        lons, lats, values, fraction)

    image = geo.stretch_image(lons, lats, values)
    IMAGES[key] = image
    return image


def _load_cube(path, variable, pts_3d, pts_4d):
    import iris
    cube = iris.load_cube(path, iris.Constraint(variable))
    units = cube.units
    lons = cube.coord('longitude').points
    if lons.ndim == 2:
        lons = lons[0, :]
    lats = cube.coord('latitude').points
    if lons.ndim == 2:
        lats = lats[:, 0]
    if cube.data.ndim == 4:
        values = cube.data[pts_4d]
    else:
        values = cube.data[pts_3d]
    return lons, lats, values, units


def _load_netcdf4(path, variable, pts_3d, pts_4d):
    with netCDF4.Dataset(path) as dataset:
        try:
            var = dataset.variables[variable]
        except KeyError as e:
            if variable == "precipitation_flux":
                var = dataset.variables["stratiform_rainfall_rate"]
            else:
                raise e
        for d in var.dimensions:
            if "longitude" in d:
                lons = dataset.variables[d][:]
            if "latitude" in d:
                lats = dataset.variables[d][:]
        if len(var.dimensions) == 4:
            values = var[pts_4d]
        else:
            values = var[pts_3d]
        units = var.units
    return lons, lats, values, units


def convert_units(values, old_unit, new_unit):
    if isinstance(values, list):
        values = np.asarray(values)
    return cf_units.Unit(old_unit).convert(values, new_unit)
