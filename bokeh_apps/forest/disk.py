"""Helpers to locate data on disk"""
import netCDF4
import datetime as dt
import numpy as np
import util


class NoData(Exception):
    pass


class GlobalUM(object):
    def __init__(self, paths):
        self.paths = np.asarray(paths)
        self.initial_times = np.array([
                util.initial_time(p) for p in paths],
                dtype='datetime64[s]')
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

    def path_index(self, initial, valid, pressure):
        path, pts = self.path_points(initial, valid, pressure)
        return path, np.argmax(pts)

    def path_points(self, initial, valid, pressure):
        if isinstance(valid, str):
            valid = np.datetime64(valid, 's')
        paths = self.run_paths(initial)
        for path in paths:
            pts = points(
                    self.valid_times[path],
                    self.pressures[path],
                    valid,
                    pressure)
            if pts.any():
                return path, pts
        raise NoData('initial: {} valid: {} pressure: {}'.format(initial, valid, pressure))

    def run_paths(self, initial):
        if isinstance(initial, str):
            initial = np.datetime64(initial, 's')
        return self.paths[self.initial_times == initial]



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
    ptol = 1
    ttol = np.timedelta64(15 * 60, 's')
    if times.dtype == 'O':
        times = times.astype('datetime64[s]')
    if isinstance(time, dt.datetime):
        time = np.datetime64(time, 's')
    return (
            (np.abs(pressures - pressure) < ptol) &
            (np.abs(times - time) < ttol))


def pressure_index(pressures, pressure):
    return np.argmin(np.abs(pressures - pressure))
