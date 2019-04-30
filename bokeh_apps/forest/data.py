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


# Application data shared across documents
FILE_DB = None
MODEL_NAMES = []
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

def on_server_loaded(patterns):
    global DISPUTED
    global COASTLINES
    global LAKES
    global BORDERS
    global FILE_DB
    global MODEL_NAMES
    FILE_DB = FileDB(patterns)
    FILE_DB.sync()
    for name, paths in FILE_DB.files.items():
        if name == "RDT":
            LOADERS[name] = rdt.Loader(paths)
        elif "GPM" in name:
            LOADERS[name] = GPM(paths)
        elif name == "EarthNetworks":
            LOADERS[name] = earth_networks.Loader(paths)
        elif name == "EIDA50":
            LOADERS[name] = satellite.EIDA50(paths)
        else:
            LOADERS[name] = UMLoader(paths, name=name)
            MODEL_NAMES.append(name)

    # Example of server-side pre-caching
    # for name in [
    #         "Tropical Africa 4.4km"]:
    #     path = FILE_DB.files[name][0]
    #     load_image(path, "relative_humidity", 0, 0)

    # Load coastlines/borders
    EXTENT = (-10, 50, -20, 10)
    COASTLINES = xs_ys(iterlines(
            cartopy.feature.COASTLINE.geometries()))
    LAKES = xs_ys(iterlines(
        at_scale(cartopy.feature.LAKES, "10m")
            .intersecting_geometries(EXTENT)))
    DISPUTED = xs_ys(iterlines(
            cartopy.feature.NaturalEarthFeature(
                "cultural",
                "admin_0_boundary_lines_disputed_areas",
                "50m").geometries()))
    BORDERS = xs_ys(iterlines(
            at_scale(cartopy.feature.BORDERS, '50m')
                .intersecting_geometries(EXTENT)))


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
    for geometry in geometries:
        for g in geometry:
            try:
                yield g.xy
            except NotImplementedError:
                if g.boundary.type == 'MultiLineString':
                    for line in g.boundary:
                        yield line.xy
                else:
                    yield g.boundary.xy


def at_scale(feature, scale):
    """
    .. note:: function named at_scale to prevent name
              clash with scale variable
    """
    feature.scale = scale
    return feature


class FileDB(object):
    def __init__(self, patterns):
        self.patterns = patterns
        self.names = list(patterns.keys())
        self.files = {}

    def sync(self):
        for key, pattern in self.patterns.items():
            self.files[key] = glob.glob(pattern)


class GPM(object):
    def __init__(self, paths):
        self.paths = paths

    def image(self, itime):
        return load_image(
                self.paths[0],
                "precipitation_flux",
                0,
                itime)


def initial_time(path):
    name = os.path.basename(path)
    groups = re.search(r"[0-9]{8}T[0-9]{4}Z", path)
    if groups:
        return dt.datetime.strptime(groups[0], "%Y%m%dT%H%MZ")


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


class UMLoader(object):
    def __init__(self, paths, name="UM"):
        self.name = name
        self.paths = paths
        self.finder = Finder(paths)
        self.initial_times = {initial_time(p): p for p in paths}
        with netCDF4.Dataset(self.paths[0]) as dataset:
            self.dimensions = self.load_dimensions(dataset)
            self.dimension_variables = self.load_dimension_variables(dataset)
            self.times = self.load_times(dataset)
            self.variables = self.load_variables(dataset)
            self.pressure_variables, self.pressures = self.load_heights(dataset)

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
    def load_heights(dataset):
        variables = set()
        pressures = dataset.variables["pressure"][:]
        for variable, var in dataset.variables.items():
            if variable == "pressure":
                continue
            if "pressure" in var.dimensions:
                variables.add(variable)
        return variables, pressures

    @staticmethod
    def load_times(dataset):
        times = {}
        for v in dataset.variables:
            var = dataset.variables[v]
            if len(var.dimensions) != 1:
                continue
            if v.startswith("time"):
                times[v] = netCDF4.num2date(
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

    def image(self, variable, ipressure, itime):
        if variable not in self.pressure_variables:
            ipressure = 0

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
        path = self.find_path(initial)
        data = load_image(
                path,
                variable,
                ipressure,
                itime)
        if variable in self.pressure_variables:
            level = "{} hPa".format(int(self.pressures[ipressure]))
        else:
            level = "Surface"
        data["name"] = [self.name]
        data["valid"] = [valid]
        data["initial"] = [initial]
        data["length"] = [length]
        data["level"] = [level]
        return data

    def find_path(self, initial_time):
        return self.finder.find(initial_time)

    def series(self, variable, x0, y0, k):
        lon0, lat0 = geo.plate_carree(x0, y0)
        lon0, lat0 = lon0[0], lat0[0]  # Map to scalar
        lons = geo.to_180(self.longitudes(variable))
        lats = self.latitudes(variable)
        i = np.argmin(np.abs(lons - lon0))
        j = np.argmin(np.abs(lats - lat0))
        return self.series_ijk(variable, i, j, k)

    def longitudes(self, variable):
        return self._lookup("longitude", variable)

    def latitudes(self, variable):
        return self._lookup("latitude", variable)

    def _lookup(self, prefix, variable):
        dims = self.dimensions[variable]
        for dim in dims:
            if dim.startswith(prefix):
                return self.dimension_variables[dim]

    def series_ijk(self, variable, i, j, k):
        path = self.paths[0]
        dimension = self.dimensions[variable][0]
        times = self.times[dimension]
        with netCDF4.Dataset(path) as dataset:
            var = dataset.variables[variable]
            if len(var.dimensions) == 4:
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


class Finder(object):
    def __init__(self, paths):
        self.paths = paths
        self.table = {
                initial_time(p): p for p in paths}

    def find(self, initial_time):
        return self.table[initial_time]


def load_image(path, variable, ipressure, itime):
    key = (path, variable, ipressure, itime)
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
                values = var[itime, ipressure, :]
            else:
                values = var[itime, :]
            # Units
            if variable in ["precipitation_flux", "stratiform_rainfall_rate"]:
                if var.units == "mm h-1":
                    values = values
                else:
                    values = convert_units(values, var.units, "kg m-2 hour-1")
            elif var.units == "K":
                values = convert_units(values, "K", "Celsius")

        # Coarsify images
        values = scipy.ndimage.zoom(values, 0.5)
        ny, nx = values.shape
        lons = np.linspace(lons.min(), lons.max(), nx)
        lats = np.linspace(lats.min(), lats.max(), ny)
        print(values.shape)
        print(lons.shape)
        print(lats.shape)

        image = geo.stretch_image(lons, lats, values)
        IMAGES[key] = image
        return image


def convert_units(values, old_unit, new_unit):
    if isinstance(values, list):
        values = np.asarray(values)
    return cf_units.Unit(old_unit).convert(values, new_unit)
