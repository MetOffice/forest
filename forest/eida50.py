import netCDF4
import numpy as np
from functools import lru_cache


class Coordinates(object):
    """Coordinate system related to EIDA50 file(s)"""
    def initial_time(self, path):
        return min(self._cached_times(path))

    def valid_times(self, path, variable):
        return self._cached_times(path)

    @lru_cache()
    def _cached_times(self, path):
        with netCDF4.Dataset(path) as dataset:
            var = dataset.variables["time"]
            values = netCDF4.num2date(var[:], units=var.units)
        return np.array(values, dtype='datetime64[s]')

    def variables(self, path):
        return ["EIDA50"]

    def pressures(self, path, variable):
        pass
