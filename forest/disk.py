"""Helpers to locate data on disk"""
import netCDF4
import datetime as dt
import numpy as np
import util
import fnmatch
import iris
from db.exceptions import SearchFail


class Navigator(object):
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
        times = [self._initial_time(p) for p in paths]
        times = [t for t in times if t is not None]
        return list(sorted(set(times)))

    def _initial_time(self, path):
        with netCDF4.Dataset(path) as dataset:
            try:
                var = dataset.variables["forecast_reference_time" ]
                return netCDF4.num2date(var[:], units=var.units)
            except KeyError:
                pass

    def valid_times(self, pattern, variable, initial_time):
        paths = fnmatch.filter(self.paths, pattern)
        arrays = []
        for path in paths:
            with netCDF4.Dataset(path) as dataset:
                t = Locator._valid_times(dataset, variable)
                if t is None:
                    cube = iris.load_cube(path, variable)
                    t = np.array([c.point for c in cube.coord('time').cells()],
                             dtype='datetime64[s]')
                elif t.ndim == 0:
                    t = np.array([t], dtype='datetime64[s]')
                arrays.append(t)
        return np.unique(np.concatenate(arrays))

    def pressures(self, pattern, variable, initial_time):
        paths = fnmatch.filter(self.paths, pattern)
        arrays = []
        for path in paths:
            with netCDF4.Dataset(path) as dataset:
                p = Locator._pressures(dataset, variable)
                if p is None:
                    cube = iris.load_cube(path, variable)
                    try:
                        p = cube.coord('pressure').points
                    except iris.exceptions.CoordinateNotFoundError:
                        return []
                elif p.ndim == 0:
                    p = np.array([p])
                arrays.append(p)
        if len(arrays) == 0:
            return []
        return np.unique(np.concatenate(arrays))


class DateLocator(object):
    def __init__(self, paths):
        self.paths = np.asarray(paths)
        self.initial_times = np.array([
                util.initial_time(p) for p in paths],
                dtype='datetime64[s]')

    def search(self, initial):
        if isinstance(initial, str):
            initial = np.datetime64(initial, 's')
        return self.paths[self.initial_times == initial]


class Locator(object):
    """Locator for collection of UM diagnostic files

    Uses file naming convention and meta-data stored in
    files to quickly look up file/index related to point
    in space/time
    """
    def __init__(self, paths=None):
        if paths is None:
            paths = []
        self.locator = DateLocator(paths)
        self.paths = np.asarray(paths)
        self.valid_times = {}
        self.pressures = {}
        for path in paths:
            with netCDF4.Dataset(path) as dataset:
                var = dataset.variables["time"]
                dates = netCDF4.num2date(var[:],
                        units=var.units)
                if isinstance(dates, dt.datetime):
                    dates = np.array([dates], dtype=object)
                self.valid_times[path] = dates.astype(
                        'datetime64[s]')
                self.pressures[path] = dataset.variables[
                        'pressure'][:]

    def search(self, *args, **kwargs):
        return self.path_points(*args, **kwargs)

    def locate(
            self,
            pattern,
            variable,
            initial_time,
            valid_time,
            pressure=None,
            tolerance=0.001):
        print(pattern, variable, initial_time, valid_time, pressure)
        raise NotImplementedError("File system search not supported")

    def path_index(self, variable, initial, valid, pressure):
        path, pts = self.path_points(variable, initial, valid, pressure)
        return path, np.argmax(pts)

    def path_points(self, variable, initial, valid, pressure):
        if isinstance(valid, str):
            valid = np.datetime64(valid, 's')
        paths = self.run_paths(initial)
        for path in paths:
            with netCDF4.Dataset(path) as dataset:
                valid_times = self._valid_times(dataset, variable)
                pressures = self._pressures(dataset, variable)
                if pressures is not None:
                    pts = points(
                            valid_times,
                            pressures,
                            valid,
                            pressure)
                else:
                    pts = time_points(
                            valid_times,
                            valid)
                if pts.any():
                    return path, pts
        raise SearchFail('initial: {} valid: {} pressure: {}'.format(initial, valid, pressure))

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

    def run_paths(self, initial):
        return self.locator.search(initial)


def points(times, pressures, time, pressure):
    """Locate slice of array for time/pressure"""
    return (
            pressure_points(pressures, pressure) &
            time_points(times, time))


def pressure_points(pressures, pressure):
    ptol = 1
    return np.abs(pressures - pressure) < ptol


def time_points(times, time):
    ttol = np.timedelta64(15 * 60, 's')
    if times.dtype == 'O':
        times = times.astype('datetime64[s]')
    if isinstance(time, dt.datetime):
        time = np.datetime64(time, 's')
    return np.abs(times - time) < ttol
