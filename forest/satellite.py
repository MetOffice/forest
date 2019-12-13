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
        data = geo.stretch_image(
                lons, lats, values)
        print(data["image"][0])
        return data


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
        print(path, time_axis, date)
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

    def find_file_index(self, paths, user_date):
        dates = np.array([
            self.parse_date(path) for path in paths],
            dtype='datetime64[s]')
        for path, _date in zip(paths, dates):
            print(path, _date)
        mask = ~(dates <= user_date)
        if mask.all():
            msg = "No file for {}".format(user_date)
            raise FileNotFound(msg)
        before_dates = np.ma.array(
                dates, mask=mask, dtype='datetime64[s]')
        return np.ma.argmax(before_dates)

    @staticmethod
    def find_index(times, time, length):
        dtype = 'datetime64[s]'
        if isinstance(times, list):
            times = np.asarray(times, dtype=dtype)
        bounds = locate.bounds(times, length)
        inside = locate.in_bounds(bounds, time)
        valid_times = np.ma.array(times, mask=~inside)
        if valid_times.mask.all():
            msg = "{}: not found".format(time)
            print(msg)
            raise IndexNotFound(msg)
        return np.ma.argmax(valid_times)

    @staticmethod
    def parse_date(path):
        groups = re.search(r"([0-9]{8})\.nc", path)
        if groups is None:
            # *20191211T0000Z.nc
            groups = re.search(r"([0-9]{8}T[0-9]{4}Z)\.nc", path)
            return dt.datetime.strptime(groups[1], "%Y%m%dT%H%MZ")
        else:
            return dt.datetime.strptime(groups[1], "%Y%m%d")
