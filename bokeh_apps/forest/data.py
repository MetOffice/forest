import os
import datetime as dt
import re
import cartopy
import glob
import json
import pandas as pd
import numpy as np
import netCDF4
import scipy.interpolate
import scipy.ndimage
import geo
from collections import OrderedDict


# Application data shared across documents
FILE_DB = None
LOADERS = {}
IMAGES = OrderedDict()
COASTLINES = {
    "xs": [],
    "ys": []
}
BORDERS = {
    "xs": [],
    "ys": []
}

def on_server_loaded(patterns):
    global COASTLINES
    global BORDERS
    global FILE_DB
    FILE_DB = FileDB(patterns)
    FILE_DB.sync()
    for name, paths in FILE_DB.files.items():
        if name == "RDT":
            LOADERS[name] = RDT(paths)
        elif "GPM" in name:
            LOADERS[name] = GPM(paths)
        elif name == "EarthNetworks":
            LOADERS[name] = EarthNetworks(paths)
        else:
            LOADERS[name] = UMLoader(paths, name=name)

    # Example of server-side pre-caching
    # for name in [
    #         "Tropical Africa 4.4km"]:
    #     path = FILE_DB.files[name][0]
    #     load_image(path, "relative_humidity", 0, 0)

    # Load coastlines/borders
    COASTLINES = feature_lines(cartopy.feature.COASTLINE)
    BORDERS = feature_lines(cartopy.feature.BORDERS)


def feature_lines(feature):
    xs, ys = [], []
    for geometry in feature.geometries():
        for g in geometry:
            lons, lats = g.xy
            x, y = geo.web_mercator(lons, lats)
            xs.append(x)
            ys.append(y)
    return {
        "xs": xs,
        "ys": ys
    }


class FileDB(object):
    def __init__(self, patterns):
        self.patterns = patterns
        self.names = list(patterns.keys())
        self.files = {}

    def sync(self):
        for key, pattern in self.patterns.items():
            self.files[key] = glob.glob(pattern)


class EarthNetworks(object):
    def __init__(self, paths):
        self.paths = paths
        self.frame = self.read(paths)
        print(self.frame)

    @staticmethod
    def read(csv_files):
        if isinstance(csv_files, str):
            csv_files = [csv_files]
        frames = []
        for csv_file in csv_files:
            frame = pd.read_csv(
                csv_file,
                parse_dates=[1],
                converters={0: EarthNetworks.flash_type},
                usecols=[0, 1, 2, 3],
                names=["flash_type", "date", "latitude", "longitude"],
                header=None)
            frames.append(frame)
        if len(frames) == 0:
            return None
        else:
            return pd.concat(frames, ignore_index=True)


    @staticmethod
    def flash_type(value):
        return {
            "0": "CG",
            "1": "IC",
            "9": "Keep alive"
        }.get(value, value)


class RDT(object):
    def __init__(self, paths):
        self.paths = paths
        self.geojson = self.load(self.paths[0])

    @staticmethod
    def load(path):
        with open(path) as stream:
            rdt = json.load(stream)

        copy = dict(rdt)
        for i, feature in enumerate(rdt["features"]):
            coordinates = feature['geometry']['coordinates'][0]
            lons, lats = np.asarray(coordinates).T
            x, y = geo.web_mercator(lons, lats)
            c = np.array([x, y]).T.tolist()
            copy["features"][i]['geometry']['coordinates'][0] = c

        # Hack to use Categorical mapper
        for i, feature in enumerate(rdt["features"]):
            p = feature['properties']['PhaseLife']
            copy["features"][i]['properties']['PhaseLife'] = str(p)

        return json.dumps(copy)


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


class UMLoader(object):
    def __init__(self, paths, name="UM"):
        self.name = name
        self.paths = paths
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
        data = load_image(
                self.paths[0],
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
        return {
            "x": times,
            "y": values}


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
        image = stretch_image(lons, lats, values)
        IMAGES[key] = image
        return image


def stretch_image(lons, lats, values):
    gx, _ = geo.web_mercator(
        lons,
        np.zeros(len(lons), dtype="d"))
    _, gy = geo.web_mercator(
        np.zeros(len(lats), dtype="d"),
        lats)
    image = stretch_y(gy)(values)
    x = gx.min()
    y = gy.min()
    dw = gx[-1] - gx[0]
    dh = gy[-1] - gy[0]
    return {
        "x": [x],
        "y": [y],
        "dw": [dw],
        "dh": [dh],
        "image": [image]
    }


def stretch_y(uneven_y):
    """Mercator projection stretches longitude spacing

    To remedy this effect an even-spaced resampling is performed
    in the projected space to make the pixels and grid line up

    .. note:: This approach assumes the grid is evenly spaced
              in longitude/latitude space prior to projection
    """
    if isinstance(uneven_y, list):
        uneven_y = np.asarray(uneven_y, dtype=np.float)
    even_y = np.linspace(
        uneven_y.min(), uneven_y.max(), len(uneven_y),
        dtype=np.float)
    index = np.arange(len(uneven_y), dtype=np.float)
    index_function = scipy.interpolate.interp1d(uneven_y, index)
    index_fractions = index_function(even_y)

    def wrapped(values, axis=0):
        if isinstance(values, list):
            values = np.asarray(values, dtype=np.float)
        assert values.ndim == 2, "Can only stretch 2D arrays"
        msg = "{} != {} do not match".format(values.shape[axis], len(uneven_y))
        assert values.shape[axis] == len(uneven_y), msg
        if axis == 0:
            i = index_fractions
            j = np.arange(values.shape[1], dtype=np.float)
        elif axis == 1:
            i = np.arange(values.shape[0], dtype=np.float)
            j = index_fractions
        else:
            raise Exception("Can only handle axis 0 or 1")
        return scipy.ndimage.map_coordinates(
            values,
            np.meshgrid(i, j, indexing="ij"),
            order=1)
    return wrapped
