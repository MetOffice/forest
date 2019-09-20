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
        for p in paths:
            try:
                times.append(Locator.initial_time(p))
            except InitialTimeNotFound:
                pass
        return list(sorted(set(times)))

    def valid_times(self, pattern, variable, initial_time):
        paths = fnmatch.filter(self.paths, pattern)
        arrays = []
        for path in paths:
            try:
                t = NetCDF4Locator.valid_times(path, variable)
            except KeyError:
                t = IrisLocator.valid_times(path, variable)
            if t is None:
                t = IrisLocator.valid_times(path, variable)
            elif t.ndim == 0:
                t = np.array([t], dtype='datetime64[s]')
            arrays.append(t)
        return np.unique(np.concatenate(arrays))

    def pressures(self, pattern, variable, initial_time):
        paths = fnmatch.filter(self.paths, pattern)
        arrays = []
        for path in paths:
            with netCDF4.Dataset(path) as dataset:
                try:
                    p = Locator._pressures(dataset, variable)
                except KeyError:
                    cube = iris.load_cube(path, variable)
                    p = self._cube_pressures(cube)
                if p is None:
                    cube = iris.load_cube(path, variable)
                    p = self._cube_pressures(cube)
                elif np.ndim(p) == 0:
                    p = np.array([p])
                arrays.append(p)
        if len(arrays) == 0:
            return []
        return np.unique(np.concatenate(arrays))

    @staticmethod
    def _cube_pressures(cube):
        try:
            return cube.coord('pressure').points
        except iris.exceptions.CoordinateNotFoundError:
            return []


class NetCDF4Locator(object):
    @staticmethod
    def initial_time(path):
        with netCDF4.Dataset(path) as dataset:
            var = dataset.variables["forecast_reference_time" ]
            values = netCDF4.num2date(var[:], units=var.units)
        return values

    @staticmethod
    def valid_times(path, variable):
        with netCDF4.Dataset(path) as dataset:
            values = NetCDF4Locator._valid_times(dataset, variable)
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


class IrisLocator(object):
    @staticmethod
    def initial_time(path):
        cubes = iris.load(path)
        if len(cubes) > 0:
            cube = cubes[0]
            return cube.coord('time').cells().next().point
        raise InitialTimeNotFound("No initial time: '{}'".format(path))

    @staticmethod
    def valid_times(path, variable):
        cube = iris.load_cube(path, variable)
        return np.array([
            c.point for c in cube.coord('time').cells()],
                 dtype='datetime64[s]')


class Locator(object):
    """Locator for collection of UM diagnostic files

    Uses file naming convention and meta-data stored in
    files to quickly look up file/index related to point
    in space/time
    """
    def __init__(self, paths):
        self.paths = {}
        for path in paths:
            key = str(self.initial_time(path))
            if key in self.paths:
                self.paths[key].append(path)
            else:
                self.paths[key] = [path]

    @staticmethod
    def initial_time(path):
        try:
            return NetCDF4Locator.initial_time(path)
        except KeyError:
            return IrisLocator.initial_time(path)

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
            valid_times = self.valid_times(path, variable)
            if self.has_pressure(path, variable):
                pressures = self.pressures(path, variable)
                t_pts = self.time_points(
                        valid_times,
                        valid_time)
                p_pts = self.pressure_points(
                        pressures,
                        pressure)
                pts = t_pts & p_pts
            else:
                pts = self.time_points(
                        valid_times,
                        valid_time)
            if pts.any():
                return path, pts
        raise SearchFail('initial: {} valid: {} pressure: {}'.format(initial_time, valid_time, pressure))

    @staticmethod
    def _pressures(dataset, variable):
        """Search dataset for pressure axis"""
        var = dataset.variables[variable]
        for d in var.dimensions:
            if d.startswith('pressure'):
                if d in dataset.variables:
                    return dataset.variables[d][:]
        coords = var.coordinates.split()
        for c in coords:
            if c.startswith('pressure'):
                return dataset.variables[c][:]

    def valid_times(self, path, variable):
        try:
            return NetCDF4Locator.valid_times(path, variable)
        except KeyError:
            return IrisLocator.valid_times(path, variable)

    @staticmethod
    def pressure_points(pressures, pressure):
        ptol = 1
        return np.abs(pressures - pressure) < ptol

    @staticmethod
    def time_points(times, time):
        ttol = np.timedelta64(15 * 60, 's')
        if times.dtype == 'O':
            times = times.astype('datetime64[s]')
        if isinstance(time, dt.datetime):
            time = np.datetime64(time, 's')
        return np.abs(times - time) < ttol
