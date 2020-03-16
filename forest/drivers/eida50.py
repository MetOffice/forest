import glob
import datetime as dt
import netCDF4
import numpy as np
from functools import lru_cache


def infinite_cache(f):
    """Unbounded cache to reduce navigation I/O

    .. note:: This information would be better saved in a database
              or file to reduce round-trips to disk
    """
    cache = {}
    def wrapped(self, path, variable):
        if path not in cache:
            cache[path] = f(self, path, variable)
        return cache[path]
    return wrapped


class Dataset:
    def __init__(self, pattern=None, **kwargs):
        self.pattern = pattern

    def navigator(self):
        return Navigator(self.pattern)


class Navigator:
    def __init__(self, pattern):
        self.pattern = pattern

    def variables(self, pattern):
        return ["EIDA50"]

    def initial_times(self, pattern, variable):
        return [dt.datetime(1970, 1, 1)]

    def valid_times(self, pattern, variable, initial_time):
        arrays = []
        for path in sorted(glob.glob(pattern)):
            arrays.append(self._valid_times(path, variable))
        if len(arrays) == 0:
            return []
        return np.unique(np.concatenate(arrays))

    @infinite_cache
    def _valid_times(self, path, variable):
        with netCDF4.Dataset(path) as dataset:
            var = dataset.variables["time"]
            values = netCDF4.num2date(var[:], units=var.units)
        return np.array(values, dtype='datetime64[s]')

    def pressures(self, pattern, variable, initial_time):
        return []
