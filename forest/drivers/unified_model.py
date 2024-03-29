import xarray
from functools import lru_cache
import os
import glob
import re
import fnmatch
import datetime as dt
import numpy as np
import netCDF4
import sqlite3
import forest.db.database
import forest.db.locate
import forest.db.health
import forest.util
import forest.map_view
import forest._profile
from forest.bases import Reusable
from forest import db, disk, geo
from forest.exceptions import (
    SearchFail,
    PressuresNotFound,
    InitialTimeNotFound,
)
from forest.drivers import gridded_forecast
import bokeh.models

try:
    import iris
except ImportError:
    # ReadTheDocs can't install iris
    pass


class NotFound(Exception):
    pass


class Sync:
    """Process to synchronize SQL database"""

    def __init__(self, database_path, pattern, directory):
        self.database_path = database_path
        self.pattern = pattern
        self.directory = directory

    def __call__(self):
        print(f"sync: {self.database_path} {self.pattern} {self.directory}")

        # Find S3 objects
        paths = glob.glob(self.full_path(self.pattern))
        s3_names = [os.path.basename(path) for path in paths]

        # Find names in database
        connection = sqlite3.connect(self.database_path)
        health_db = forest.db.health.HealthDB(connection)
        sql_names = [
            os.path.basename(path)
            for path in health_db.checked_files(self.pattern)
        ]
        connection.close()

        # Find extra files
        extra_names = set(s3_names) - set(sql_names)
        extra_paths = [self.full_path(name) for name in extra_names]

        # Add NetCDF files to database
        if len(extra_paths) > 0:
            print("connecting to: {}".format(self.database_path))
            with forest.db.database.Database.connect(
                self.database_path
            ) as database:
                health_db = forest.db.health.HealthDB(database.connection)
                for path in extra_paths:
                    print("inserting: '{}'".format(path))
                    try:
                        database.insert_netcdf(path)
                    except OSError as e:
                        # S3 Glacier objects inaccessible via goofys
                        health_db.insert_error(path, e, dt.datetime.now())
                        print(e)
                        print(f"skip file: {path}")
                        continue
            print("finished")

    def full_path(self, name):
        """Prepend directory if available"""
        if self.directory is None:
            return name
        else:
            return os.path.join(self.directory, name)


class Facade:
    """Translation layer between dataset labels and file system patterns
    """

    def __init__(self, pattern, database):
        self.pattern = pattern
        self.database = database

    def variables(self, _label):
        return self.database.variables(self.pattern)

    def initial_times(self, _label, variable):
        return self.database.initial_times(self.pattern, variable)

    def valid_times(self, _label, variable, initial_time):
        return self.database.valid_times(self.pattern, variable, initial_time)

    def pressures(self, _label, variable, initial_time):
        return self.database.pressures(self.pattern, variable, initial_time)


class Dataset:
    def __init__(
        self,
        label=None,
        pattern=None,
        locator="file_system",
        directory=None,
        database_path=None,
        **kwargs,
    ):
        self.label = label
        self.pattern = pattern
        self.use_database = locator == "database"
        if self.use_database:
            self.sync = Sync(database_path, pattern, directory)
            self.database = db.get_database(database_path)
            self.locator = forest.db.locate.Locator(
                self.database.connection, directory=directory
            )
        else:
            self.locator = Locator.pattern(self.pattern)

    def navigator(self):
        if self.use_database:
            return Facade(self.pattern, self.database)
        else:
            return Navigator(self.pattern)

    def map_view(self, color_mapper=None):
        loader = Loader(self.label, self.pattern, self.locator)
        return forest.map_view.map_view(loader, color_mapper)

    def profile_view(self, figure):
        loader = Loader(self.label, self.pattern, self.locator)
        return ProfileView(figure, loader)

    def series_view(self, figure):
        return SeriesView(figure)


class ProfileView(Reusable):
    def __init__(self, figure, loader):
        self.figure = figure
        self.loader = loader
        self.source = bokeh.models.ColumnDataSource({"x": [], "y": []})
        self.renderers = [
            self.figure.line(x="x", y="y", source=self.source),
            self.figure.circle(x="x", y="y", source=self.source),
        ]

    def prepare(self):
        for renderer in self.renderers:
            renderer.visible = True

    def reset(self):
        for renderer in self.renderers:
            renderer.visible = False
        self.source.data = {"x": [], "y": []}

    def render_id(self, state, layer_id):
        print(f"{self.__class__.__name__}.render({layer_id})")
        lons, lats = geo.plate_carree(state.position.x, state.position.y)
        lon_0 = lons[0]
        lat_0 = lats[0]
        self.source.data = self.loader.profile(
            state.pattern,
            state.layers.index[layer_id]["variable"],
            state.initial_time,
            state.valid_time,
            state.pressure,
            lon_0,
            lat_0,
        )


class SeriesView(Reusable):
    def __init__(self, figure):
        self.figure = figure
        self.source = bokeh.models.ColumnDataSource({"x": [], "y": []})
        self.renderers = [
            self.figure.line(x="x", y="y", source=self.source),
            self.figure.circle(x="x", y="y", source=self.source),
        ]

    def prepare(self):
        for renderer in self.renderers:
            renderer.visible = True

    def reset(self):
        for renderer in self.renderers:
            renderer.visible = False
        self.source.data = {"x": [], "y": []}

    def render_id(self, state, layer_id):
        print(f"{self.__class__.__name__}.render({layer_id})")
        self.source.data = {
            "x": [0, 1, 2],
            "y": [0, layer_id, 2 * layer_id],
        }


class Navigator:
    def __init__(self, pattern):
        self.pattern = pattern
        self._locators = {
            "initial": read_initial_time,
            "valid": read_valid_times,
            "pressure": PressuresLocator(),
        }

    def variables(self, _pattern):
        cubes = iris.load(self.pattern)
        return [cube.name() for cube in cubes]

    def initial_times(self, _pattern, variable):
        locator = self._locators["initial"]
        return list(
            sorted(set(locator(path) for path in glob.glob(self.pattern)))
        )

    def valid_times(self, _pattern, variable, initial_time):
        return self._dimension("valid", self.pattern, variable, initial_time)

    def pressures(self, _pattern, variable, initial_time):
        return self._dimension(
            "pressure", self.pattern, variable, initial_time
        )

    def _dimension(self, keyword, pattern, variable, initial_time):
        arrays = []
        locator = self._locators[keyword]
        for path in glob.glob(pattern):
            arrays.append(locator(path, variable))
        if len(arrays) == 0:
            return []
        return np.unique(np.concatenate(arrays))


class Loader:
    """Unified model formatted loader"""

    def __init__(self, name, pattern, locator):
        self.name = name
        self.pattern = pattern
        self.locator = locator

    def image(self, state):
        if not self.valid(state):
            return gridded_forecast.empty_image()
        data = self._input_output(
            self.pattern,
            state.variable,
            state.initial_time,
            state.valid_time,
            state.pressure,
        )
        data.update(
            gridded_forecast.coordinates(
                state.valid_time,
                state.initial_time,
                state.pressures,
                state.pressure,
            )
        )
        return data

    def profile(
        self,
        pattern,
        variable,
        initial_time,
        valid_time,
        pressure,
        lon_0,
        lat_0,
    ):
        """Load Profile from file"""
        try:
            path, _ = self.locator.locate(
                pattern, variable, initial_time, valid_time, pressure
            )
        except SearchFail:
            return {"x": [], "y": []}

        with xarray.open_dataset(path, engine="h5netcdf") as nc:
            data_array = nc[variable]
            print(data_array.shape)
            lons = np.ma.masked_invalid(data_array.longitude)
            lats = np.ma.masked_invalid(data_array.latitude)
            i = np.argmin(np.abs(lons - lon_0))
            j = np.argmin(np.abs(lats - lat_0))

            # Generalised profile slice needed
            y = np.ma.masked_invalid(data_array.dim0)
            x = np.ma.masked_invalid(data_array[:, i, j])

            # Convert NaN to masked
            x = np.ma.masked_array(x, mask=np.isnan(x))
            y = np.ma.masked_array(y, mask=np.isnan(y))

        return {
            "x": x,
            "y": y,
        }

    @lru_cache(maxsize=100)
    def _input_output(
        self, pattern, variable, initial_time, valid_time, pressure
    ):
        """I/O needed to load an image and its metadata"""
        try:
            path, pts = self.locator.locate(
                pattern, variable, initial_time, valid_time, pressure
            )
        except SearchFail:
            return gridded_forecast.empty_image()

        data = self.load_image(path, variable, pts)
        data["name"] = [self.name]
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
            if state.pressure is None:
                return False
            if not self.has_pressure(state.pressures, state.pressure):
                return False
        return True

    def has_pressure(self, pressures, pressure, tolerance=0.01):
        if isinstance(pressures, list):
            pressures = np.array(pressures)
        return any(np.abs(pressures - pressure) < tolerance)

    @classmethod
    def load_image(cls, path, variable, pts):
        """Load bokeh image glyph data from file using slices"""
        try:
            lons, lats, values, units = cls._load_xarray(path, variable, pts)
        except:
            lons, lats, values, units = cls._load_cube(path, variable, pts)

        # Units
        if variable in ["precipitation_flux", "stratiform_rainfall_rate"]:
            if units == "mm h-1":
                values = values
            else:
                values = forest.util.convert_units(
                    values, units, "kg m-2 hour-1"
                )
                units = "kg m-2 hour-1"
        elif units == "K":
            values = forest.util.convert_units(values, "K", "Celsius")
            units = "C"

        # Coarsify images
        threshold = 200 * 200  # Chosen since TMA WRF is 199 x 199
        if values.size > threshold:
            fraction = 0.25
        else:
            fraction = 1.0
        lons, lats, values = forest.util.coarsify(lons, lats, values, fraction)

        # Roll input data into [-180, 180] range
        if np.any(lons > 180.0):
            shift_by = np.sum(lons > 180.0)
            lons[lons > 180.0] -= 360.0
            lons = np.roll(lons, shift_by)
            values = np.roll(values, shift_by, axis=1)

        data = geo.stretch_image(lons, lats, values)
        data["units"] = [units]
        return data

    @staticmethod
    def _load_xarray(path, variable, pts):
        with xarray.open_dataset(path, engine="h5netcdf") as nc:
            data_array = nc[variable][pts]
            lons = np.ma.masked_invalid(data_array.longitude)
            lats = np.ma.masked_invalid(data_array.latitude)
            values = np.ma.masked_invalid(data_array)
            units = getattr(data_array, "units", "")
        return lons, lats, values, units

    @staticmethod
    def _load_cube(path, variable, pts):
        # TODO: Is this method still needed?
        cube = iris.load_cube(path, iris.Constraint(variable))
        units = cube.units
        lons = cube.coord("longitude").points
        if lons.ndim == 2:
            lons = lons[0, :]
        lats = cube.coord("latitude").points
        if lons.ndim == 2:
            lats = lats[:, 0]
        values = cube.data[pts]
        return lons, lats, values, str(units)  # Needed for tutorial data


class Locator(object):
    def __init__(self, paths):
        self.paths = paths
        self.spare = []
        self.catalogue = {}
        for path in paths:
            initial_time = self.initial_time(path)
            if initial_time is None:
                self.spare.append(path)
                continue
            key = self.key(initial_time)
            if key not in self.catalogue:
                self.catalogue[key] = [path]
            else:
                self.catalogue[key].append(path)

    @classmethod
    def pattern(cls, text):
        return cls(sorted(glob.glob(os.path.expanduser(text))))

    def locate(
        self,
        pattern,
        variable,
        initial_time,
        valid_time,
        pressure=None,
        tolerance=0.001,
    ):
        paths = self.find_paths(initial_time) + self.spare
        paths = fnmatch.filter(paths, pattern)
        for path in paths:
            with netCDF4.Dataset(path) as dataset:
                if variable not in dataset.variables:
                    continue

                var = dataset.variables[variable]
                dims = var.dimensions
                coords = getattr(var, "coordinates", "")

                masks = {}
                for coord, value in [
                    ("time", valid_time),
                    ("pressure", pressure),
                ]:
                    if not disk.has_coord(coord, dims, coords):
                        continue
                    if value is None:
                        # Coordinate present but value not specified
                        raise SearchFail("Please specify: '{}'".format(coord))
                        continue
                    axis = disk.axis(coord, dims, coords)
                    coord_var = disk.coord_var(coord, dims, coords)
                    if coord == "time":
                        obj = dataset.variables[coord_var]
                        values = netCDF4.num2date(obj[:], units=obj.units)
                    else:
                        values = dataset.variables[coord_var][:]
                    mask = disk.coord_mask(coord, values, value)
                    if axis not in masks:
                        masks[axis] = mask
                    else:
                        masks[axis] = masks[axis] & mask

            # Determine if search was successful
            found = all(mask.any() for mask in masks.values())
            if not found:
                continue

            # Generate multi-dimensional slice from search result
            slices = []
            for i in range(max(masks.keys()) + 1):
                pts = np.where(masks[i])[0][0]
                slices.append(pts)
            pts = tuple(slices)
            return path, pts

        # Search failure message
        msg = " ".join(
            [
                str(value)
                for value in [
                    pattern,
                    variable,
                    initial_time,
                    valid_time,
                    pressure,
                ]
            ]
        )
        raise SearchFail(msg)

    def find_paths(self, initial_time):
        return self.catalogue.get(self.key(initial_time), [])

    @staticmethod
    def key(time):
        if isinstance(time, str):
            from dateutil import parser

            time = parser.parse(time)
        return time.strftime("%Y%m%dT%H%M%S")

    def initial_time(self, path):
        for strategy in [self.initial_time_regex, self.initial_time_netcdf4]:
            result = strategy(path)
            if result is None:
                continue
            else:
                return result

    def initial_time_regex(self, path):
        name = os.path.basename(path)
        groups = re.search(r"[0-9]{8}T[0-9]{4}Z", path)
        if groups:
            return dt.datetime.strptime(groups[0], "%Y%m%dT%H%MZ")

    def initial_time_netcdf4(self, path):
        with netCDF4.Dataset(path) as dataset:
            try:
                var = dataset.variables["forecast_reference_time"]
                result = netCDF4.num2date(var[:], units=var.units)
            except KeyError:
                result = None
        return result


def read_initial_time(path):
    return InitialTimeLocator()(path)


class InitialTimeLocator:
    def __call__(self, path):
        try:
            return self.netcdf4_strategy(path)
        except KeyError:
            return self.cube_strategy(path)

    @staticmethod
    def netcdf4_strategy(path):
        with netCDF4.Dataset(path) as dataset:
            var = dataset.variables["forecast_reference_time"]
            values = netCDF4.num2date(var[:], units=var.units)
        return values

    @staticmethod
    def cube_strategy(path):
        cubes = iris.load(path)
        if len(cubes) > 0:
            cube = cubes[0]
            return cube.coord("time").cells().next().point
        raise InitialTimeNotFound("No initial time: '{}'".format(path))


def read_valid_times(path, variable):
    return ValidTimesLocator()(path, variable)


class ValidTimesLocator(object):
    def __call__(self, path, variable):
        try:
            t = self.netcdf4_strategy(path, variable)
        except KeyError:
            t = self.cube_strategy(path, variable)
        if t is None:
            t = self.cube_strategy(path, variable)
        elif t.ndim == 0:
            t = np.array([t], dtype="datetime64[s]")
        return t

    def netcdf4_strategy(self, path, variable):
        with netCDF4.Dataset(path) as dataset:
            values = self._valid_times(dataset, variable)
        return values

    @staticmethod
    def _valid_times(dataset, variable):
        """Search dataset for time axis"""
        var = dataset.variables[variable]
        for d in var.dimensions:
            if d.startswith("time"):
                if d in dataset.variables:
                    tvar = dataset.variables[d]
                    return np.array(
                        netCDF4.num2date(tvar[:], units=tvar.units),
                        dtype="datetime64[s]",
                    )
        coords = var.coordinates.split()
        for c in coords:
            if c.startswith("time"):
                tvar = dataset.variables[c]
                return np.array(
                    netCDF4.num2date(tvar[:], units=tvar.units),
                    dtype="datetime64[s]",
                )

    @staticmethod
    def cube_strategy(path, variable):
        cube = iris.load_cube(path, variable)
        return np.array(
            [c.point for c in cube.coord("time").cells()],
            dtype="datetime64[s]",
        )


class PressuresLocator(object):
    def __call__(self, path, variable):
        try:
            return self.netcdf4_strategy(path, variable)
        except KeyError:
            return self.cube_strategy(path, variable)

    def cube_strategy(self, path, variable):
        try:
            cube = iris.load_cube(path, variable)
            points = cube.coord("pressure").points
            if np.ndim(points) == 0:
                points = np.array([points])
            return points
        except iris.exceptions.CoordinateNotFoundError:
            raise PressuresNotFound("'{}' '{}'".format(path, variable))

    @staticmethod
    def netcdf4_strategy(path, variable):
        """Search dataset for pressure axis"""
        with netCDF4.Dataset(path) as dataset:
            var = dataset.variables[variable]
            for d in var.dimensions:
                if d.startswith("pressure"):
                    if d in dataset.variables:
                        return dataset.variables[d][:]
            coords = var.coordinates.split()
            for c in coords:
                if c.startswith("pressure"):
                    return dataset.variables[c][:]
        # NOTE: refactor needed
        raise KeyError
