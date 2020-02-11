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


class Coordinates(object):
    """Coordinate system related to EIDA50 file(s)"""
    def initial_time(self, path):
        return dt.datetime(1970, 1, 1)  # Placeholder for missing dimension

    @infinite_cache
    def valid_times(self, path, variable):
        with netCDF4.Dataset(path) as dataset:
            var = dataset.variables["time"]
            values = netCDF4.num2date(var[:], units=var.units)
        return np.array(values, dtype='datetime64[s]')

    def variables(self, path):
        return ["EIDA50"]

    def pressures(self, path, variable):
        return []
