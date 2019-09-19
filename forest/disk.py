"""Helpers to locate data on disk"""
import netCDF4
import datetime as dt
import numpy as np
import util
from db.exceptions import SearchFail


class Navigator(object):
    def __init__(self, paths):
        self.paths = paths

    def variables(self, pattern):
        return ['air_temperature']

    def initial_times(self, pattern, variable=None):
        return ['2019-01-01 00:00:00']

    def valid_times(self, pattern, variable, initial_time):
        return ['2019-01-01 12:00:00']

    def pressures(self, pattern, variable, initial_time):
        return [750.]


def scrape(path):
    """All meta-data in single transaction"""
    scheme = {
        'dimensions': [],
        'variables': {}
    }
    with netCDF4.Dataset(path) as dataset:
        for d in dataset.dimensions:
            scheme['dimensions'].append(d)
        for v, obj in dataset.variables.items():
            scheme['variables'][v] = {
                'dimensions': obj.dimensions,
                'attrs': {a: getattr(obj, a) for a in obj.ncattrs()}
            }
    return scheme


def coordinate_variables(scheme):
    """In-memory coordinate variable detection"""
    coords = set()
    for item in scheme.items():
        attrs = item['attrs']
        if 'coordinates' not in attrs:
            continue
        for c in attrs['coordinates'].split():
            coords.add(c)
    return coords


def dimension_variables(scheme):
    """In-memory dimension variable detection"""
    names = set()
    for item in scheme.items():
        for d in scheme['dimensions']:
            if d in scheme['variables']:
                names.add(d)
    return names


def load_variables(path, names):
    values = {}
    with netCDF4.Dataset(path) as dataset:
        for name in names:
            try:
                var = dataset.variables[name]
            except KeyError:
                continue
            if 'time' in name:
                datetimes = netCDF4.num2date(var[:], units=var.units)
                values[name] = np.array(datetimes, dtype='datetime64[s]')
            else:
                values[name] = var[:]
    return values


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
        raise SearchFail

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



def file_name(pattern, initial, length):
    if isinstance(initial, np.datetime64):
        initial = initial.astype(dt.datetime)
    if isinstance(length, np.timedelta64):
        length = length / np.timedelta64(1, 'h')
    return pattern.format(initial, int(length))


def lengths(times, initial):
    if isinstance(initial, dt.datetime):
        initial = np.datetime64(initial, 's')
    if times.dtype == 'O':
        times = times.astype('datetime64[s]')
    return (times - initial) / np.timedelta64(1, 'h')


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


def pressure_index(pressures, pressure):
    return np.argmin(np.abs(pressures - pressure))
