import os
import datetime as dt
import re
import cartopy
import glob
import json
import pandas as pd
import numpy as np
import netCDF4
import cf_units
import satellite
import rdt
import earth_networks
import geo
import bokeh.models
from collections import OrderedDict
from functools import partial
import scipy.ndimage
import shapely.geometry
from util import timeout_cache, initial_time, coarsify
from db.exceptions import SearchFail
import disk


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


def file_loader(file_type, pattern):
    if file_type.lower() == 'rdt':
        return rdt.Loader(pattern)
    elif file_type.lower() == 'gpm':
        return GPM(pattern)
    elif file_type.lower() == 'earthnetworks':
        return earth_networks.Loader(pattern)
    elif file_type.lower() == 'eida50':
        return satellite.EIDA50(pattern)


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
                print("load from disk")
                store[args] = f(*args)
            else:
                print("seen before")
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
        print("load_data", path, ipressure, itime)
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
        self.empty_image = {
            "x": [],
            "y": [],
            "dw": [],
            "dh": [],
            "image": [],
            "name": [],
            "valid": [],
            "initial": [],
            "length": [],
            "level": []
        }

    def image(self, state):
        if not self.valid(state):
            return self.empty_image

        try:
            path, pts = self.locator.locate(
                self.pattern,
                state.variable,
                state.initial_time,
                state.valid_time,
                state.pressure)
        except SearchFail:
            return self.empty_image

        valid = dt.datetime.strptime(state.valid_time, "%Y-%m-%d %H:%M:%S")
        initial = dt.datetime.strptime(state.initial_time, "%Y-%m-%d %H:%M:%S")
        hours = (valid - initial).total_seconds() / (60*60)
        length = "T{:+}".format(int(hours))
        data = load_image_pts(
                path,
                state.variable,
                pts,
                pts)
        if (len(state.pressures) > 0) and (state.pressure is not None):
            level = "{} hPa".format(int(state.pressure))
        else:
            level = "Surface"
        data["name"] = [self.name]
        data["valid"] = [valid]
        data["initial"] = [initial]
        data["length"] = [length]
        data["level"] = [level]
        return data

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
            if not self.has_pressure(state.pressures, state.pressure):
                return False
        return True

    def has_pressure(self, pressures, pressure, tolerance=0.01):
        if isinstance(pressures, list):
            pressures = np.array(pressures)
        return any(np.abs(pressures - pressure) < tolerance)


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

    @classmethod
    def from_pattern(cls, pattern):
        return cls(sorted(glob.glob(os.path.expanduser(pattern))))

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

    def series(self, variable, x0, y0, pressure):
        if len(self.paths) > 0:
            path = self.paths[-1]
        if path is None:
            return {"x": [], "y": []}
        lon0, lat0 = geo.plate_carree(x0, y0)
        lon0, lat0 = lon0[0], lat0[0]  # Map to scalar
        lons = geo.to_180(self.longitudes(variable))
        lats = self.latitudes(variable)
        i = np.argmin(np.abs(lons - lon0))
        j = np.argmin(np.abs(lats - lat0))
        dimension = self.dimensions[variable][0]
        with netCDF4.Dataset(path) as dataset:
            times = self._times(dataset, dimension)
            pressures = self._pressures(dataset, dimension)
            pts = self.search(pressures, pressure)
            times = times[pts]
            var = dataset.variables[variable]
            if var.dimensions[0] == dimension:
                values = var[:, j, i]
                values = values[pts]
            elif len(var.dimensions) == 4:
                values = var[:, k, j, i]
            elif len(var.dimensions) == 3:
                values = var[:, j, i]
            else:
                raise NotImplementedError("3 or 4 dimensions only")
            if var.units == "K":
                values = convert_units(values, "K", "Celsius")
        return {
            "x": times,
            "y": values}

    @staticmethod
    def _times(dataset, dimension):
        for v, var in dataset.variables.items():
            if len(var.dimensions) != 1:
                continue
            if v.startswith("time"):
                d = var.dimensions[0]
                if d == dimension:
                    return netCDF4.num2date(var[:], units=var.units)

    @staticmethod
    def _pressures(dataset, dimension):
        for v, var in dataset.variables.items():
            if len(var.dimensions) != 1:
                continue
            if v.startswith("pressure"):
                d = var.dimensions[0]
                if d == dimension:
                    return var[:]

    @staticmethod
    def search(pressures, pressure, rtol=0.01):
        return np.abs(pressures - pressure) < (rtol * pressure)

    def longitudes(self, variable):
        return self._lookup("longitude", variable)

    def latitudes(self, variable):
        return self._lookup("latitude", variable)

    def _lookup(self, prefix, variable):
        dims = self.dimensions[variable]
        for dim in dims:
            if dim.startswith(prefix):
                return self.dimension_variables[dim]


class Finder(object):
    def __init__(self, paths):
        self.paths = paths
        self.table = {
                initial_time(p): p for p in paths}
        self.initial_times = np.array(
                [initial_time(p) for p in paths],
                dtype='datetime64[s]')
        with netCDF4.Dataset(self.paths[0]) as dataset:
            self.pressure_variables, self.pressures = self.load_heights(dataset)

    @staticmethod
    def load_heights(dataset):
        variables = set()
        pressures = dataset.variables["pressure"][:]
        for variable, var in dataset.variables.items():
            if variable == "pressure":
                continue
            if "pressure" in var.dimensions:
                variables.add(variable)
        return variables, pressures

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
        print("already seen: {}".format(key))
        return IMAGES[key]
    else:
        print("loading: {}".format(key))
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
            # Units
            if variable in ["precipitation_flux", "stratiform_rainfall_rate"]:
                if var.units == "mm h-1":
                    values = values
                else:
                    values = convert_units(values, var.units, "kg m-2 hour-1")
            elif var.units == "K":
                values = convert_units(values, "K", "Celsius")

        # DEBUG
        print(pts_3d, values.shape)

        # Coarsify images
        fraction = 0.25
        lons, lats, values = coarsify(
                lons, lats, values, fraction)

        image = geo.stretch_image(lons, lats, values)
        IMAGES[key] = image
        return image


def convert_units(values, old_unit, new_unit):
    if isinstance(values, list):
        values = np.asarray(values)
    return cf_units.Unit(old_unit).convert(values, new_unit)
