"""Helpers to locate data on disk"""
import netCDF4
import datetime as dt
import numpy as np
from forest import util
import fnmatch
import iris
from forest.db.exceptions import SearchFail


class InitialTimeNotFound(Exception):
    pass


class ValidTimesNotFound(Exception):
    pass


class PressuresNotFound(Exception):
    pass


class AxisNotFound(Exception):
    pass


def ndindex(masks, axes):
    """N-dimensional array indexing

    Given logical masks and their axes generate
    a multi-dimensional slice

    :returns: tuple(slices)
    """
    joint = {}
    for mask, axis in zip(masks, axes):
        print(mask, axis)
        if axis in joint:
            joint[axis] = joint[axis] & mask
        else:
            joint[axis] = mask
    slices = []
    for i in range(max(joint.keys()) + 1):
        pts = np.where(joint[i])[0][0]
        slices.append(pts)
    return tuple(slices)


def time_mask(times, time):
    """Logical mask that selects particular time"""
    if isinstance(time, (str, dt.datetime)):
        time = np.datetime64(time, 's')
    if isinstance(times, list):
        times = np.array(times, dtype='datetime64[s]')
    return times == time


def pressure_mask(pressures, pressure, rtol=0.01):
    """Logical mask that selects particular pressure"""
    if isinstance(pressures, list):
        pressures = np.array(pressures, dtype='d')
    return (np.abs(pressures - pressure) / np.abs(pressure)) < rtol


def pressure_axis(path, variable):
    return _axis("pressure", path, variable)


def time_axis(path, variable):
    return _axis("time", path, variable)


def _axis(name, path, variable):
    with netCDF4.Dataset(path) as dataset:
        var = dataset.variables[variable]
        for i, d in enumerate(var.dimensions):
            if d.startswith(name):
                return i
        coords = var.coordinates.split()
        for c in coords:
            if c.startswith(name):
                return 0
    msg = "{} axis not found: '{}' '{}'".format(name.capitalize(), path, variable)
    raise AxisNotFound(msg)


class Navigator(object):
    """Menu system given unified model files"""
    def __init__(self, paths):
        self.paths = paths

    def variables(self, pattern):
        paths = fnmatch.filter(self.paths, pattern)
        names = []
        for path in paths:
            cubes = iris.load(path)
            names += [cube.name() for cube in cubes]
        return list(sorted(set(names)))

    def initial_times(self, pattern, variable=None):
        paths = fnmatch.filter(self.paths, pattern)
        times = []
        for path in paths:
            try:
                times.append(load_initial_time(path))
            except InitialTimeNotFound:
                pass
        return list(sorted(set(times)))

    def valid_times(self, pattern, variable, initial_time):
        paths = fnmatch.filter(self.paths, pattern)
        arrays = []
        for path in paths:
            try:
                arrays.append(load_valid_times(path, variable))
            except ValidTimesNotFound:
                pass
        if len(arrays) == 0:
            return []
        return np.unique(np.concatenate(arrays))

    def pressures(self, pattern, variable, initial_time):
        paths = fnmatch.filter(self.paths, pattern)
        arrays = []
        for path in paths:
            try:
                arrays.append(load_pressures(path, variable))
            except PressuresNotFound:
                pass
        if len(arrays) == 0:
            return []
        return np.unique(np.concatenate(arrays))


class InitialTimeLocator(object):
    def __call__(self, path):
        try:
            return self.netcdf4_strategy(path)
        except KeyError:
            return self.cube_strategy(path)

    @staticmethod
    def netcdf4_strategy(path):
        with netCDF4.Dataset(path) as dataset:
            var = dataset.variables["forecast_reference_time" ]
            values = netCDF4.num2date(var[:], units=var.units)
        return values

    @staticmethod
    def cube_strategy(path):
        cubes = iris.load(path)
        if len(cubes) > 0:
            cube = cubes[0]
            return cube.coord('time').cells().next().point
        raise InitialTimeNotFound("No initial time: '{}'".format(path))


class ValidTimesLocator(object):
    def __call__(self, path, variable):
        try:
            t = self.netcdf4_strategy(path, variable)
        except KeyError:
            t = self.cube_strategy(path, variable)
        if t is None:
            t = self.cube_strategy(path, variable)
        elif t.ndim == 0:
            t = np.array([t], dtype='datetime64[s]')
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
            if d.startswith('time'):
                if d in dataset.variables:
                    tvar = dataset.variables[d]
                    return np.array(
                        netCDF4.num2date(tvar[:], units=tvar.units),
                        dtype='datetime64[s]')
        coords = var.coordinates.split()
        for c in coords:
            if c.startswith('time'):
                tvar = dataset.variables[c]
                return np.array(
                    netCDF4.num2date(tvar[:], units=tvar.units),
                    dtype='datetime64[s]')

    @staticmethod
    def cube_strategy(path, variable):
        cube = iris.load_cube(path, variable)
        return np.array([
            c.point for c in cube.coord('time').cells()],
                 dtype='datetime64[s]')


class PressuresLocator(object):
    def __call__(self, path, variable):
        try:
            return self.netcdf4_strategy(path, variable)
        except KeyError:
            return self.cube_strategy(path, variable)

    def cube_strategy(self, path, variable):
        try:
            cube = iris.load_cube(path, variable)
            points = cube.coord('pressure').points
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
                if d.startswith('pressure'):
                    if d in dataset.variables:
                        return dataset.variables[d][:]
            coords = var.coordinates.split()
            for c in coords:
                if c.startswith('pressure'):
                    return dataset.variables[c][:]
        # NOTE: refactor needed
        raise KeyError

load_pressures = PressuresLocator()
load_valid_times = ValidTimesLocator()
load_initial_time = InitialTimeLocator()


class Locator(object):
    """Locator for collection of UM diagnostic files

    Uses file naming convention and meta-data stored in
    files to quickly look up file/index related to point
    in space/time
    """
    def __init__(self, paths):
        self.paths = {}
        for path in paths:
            key = str(load_initial_time(path))
            if key in self.paths:
                self.paths[key].append(path)
            else:
                self.paths[key] = [path]

    def locate(
            self,
            pattern,
            variable,
            initial_time,
            valid_time,
            pressure=None,
            tolerance=0.001):
        if isinstance(valid_time, str):
            valid_time = np.datetime64(valid_time, 's')
        paths = self.paths[str(initial_time)]
        paths = fnmatch.filter(paths, pattern)
        for path in paths:
            valid_times = load_valid_times(path, variable)
            axes = [time_axis(path, variable)]
            masks = [time_mask(valid_times, valid_time)]
            try:
                pressures = load_pressures(path, variable)
                axis = pressure_axis(path, variable)
                masks.append(pressure_mask(pressures, pressure))
                axes.append(axis)
            except PressuresNotFound:
                pass
            pts = ndindex(masks, axes)
            print(path, pts)
            return path, pts
        raise SearchFail('initial: {} valid: {} pressure: {}'.format(
            initial_time, valid_time, pressure))
