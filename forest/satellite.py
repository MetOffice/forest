import re
import datetime as dt
import netCDF4
import numpy as np
import os
import glob
from functools import lru_cache
from forest import (
        geo,
        locate)
from forest.util import coarsify
from forest.exceptions import FileNotFound, IndexNotFound


class EIDA50(object):
    def __init__(self, pattern):
        self.locator = Locator(pattern)
        self.cache = {}
        paths = self.locator.paths()
        if len(paths) > 0:
            with netCDF4.Dataset(paths[-1]) as dataset:
                self.cache["longitude"] = dataset.variables["longitude"][:]
                self.cache["latitude"] = dataset.variables["latitude"][:]

    @property
    def longitudes(self):
        return self.cache["longitude"]

    @property
    def latitudes(self):
        return self.cache["latitude"]

    def image(self, valid_time):
        path, itime = self.locator.find(valid_time)
        return self.load_image(path, itime)

    def load_image(self, path, itime):
        lons = self.longitudes
        lats = self.latitudes
        with netCDF4.Dataset(path) as dataset:
            values = dataset.variables["data"][itime]
        fraction = 0.25
        lons, lats, values = coarsify(
                lons, lats, values, fraction)
        return geo.stretch_image(
                lons, lats, values)


class Locator(object):
    """Locate EIDA50 satellite images"""
    def __init__(self, pattern):
        self.pattern = pattern

    def find(self, date):
        if isinstance(date, (dt.datetime, str)):
            date = np.datetime64(date, 's')
        paths = self.paths()
        ipath = self.find_file_index(paths, date)
        path = paths[ipath]
        time_axis = self.load_time_axis(path)
        index = self.find_index(
                time_axis,
                date,
                dt.timedelta(minutes=15))
        return path, index

    def paths(self):
        return sorted(glob.glob(os.path.expanduser(self.pattern)))

    @staticmethod
    @lru_cache()
    def load_time_axis(path):
        with netCDF4.Dataset(path) as dataset:
            var = dataset.variables["time"]
            values = netCDF4.num2date(
                    var[:], units=var.units)
        return np.array(values, dtype='datetime64[s]')

    def find_file_index(self, paths, date):
        dates = np.array([
            self.parse_date(path) for path in paths],
            dtype='datetime64[s]')
        mask = ~(dates <= date)
        if mask.all():
            msg = "No file for {}".format(date)
            raise FileNotFound(msg)
        before_dates = np.ma.array(
                dates, mask=mask, dtype='datetime64[s]')
        return np.ma.argmax(before_dates)

    def find_index(self, times, time, length):
        dtype = 'datetime64[s]'
        if isinstance(times, list):
            times = np.asarray(times, dtype=dtype)
        bounds = locate.bounds(times, length)
        inside = locate.in_bounds(bounds, time)
        valid_times = np.ma.array(times, mask=~inside)
        if valid_times.mask.all():
            msg = "{}: not found".format(time)
            raise IndexNotFound(msg)
        return np.ma.argmax(valid_times)

    @staticmethod
    def parse_date(path):
        groups = re.search(r"([0-9]{8})\.nc", path)
        return dt.datetime.strptime(groups[1], "%Y%m%d")
