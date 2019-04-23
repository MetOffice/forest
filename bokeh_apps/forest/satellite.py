import re
import datetime as dt
import netCDF4
import numpy as np
import os
import geo


class EIDA50(object):
    total_seconds = np.vectorize(dt.timedelta.total_seconds)
    def __init__(self, paths):
        self.paths = [
                os.path.expanduser(p) for p in paths]
        self.dates = [
            self.parse_date(path) for path in paths]
        self.cache = {}
        with netCDF4.Dataset(self.paths[-1]) as dataset:
            self.cache["longitude"] = dataset.variables["longitude"][:]
            self.cache["latitude"] = dataset.variables["latitude"][:]
            var = dataset.variables["time"]
            times = netCDF4.num2date(var[:], units=var.units)
            self.cache[(self.paths[-1], "time")] = times

    @staticmethod
    def parse_date(path):
        groups = re.search(r"_([0-9]{8}).nc", path)
        return dt.datetime.strptime(groups[1], "%Y%m%d")

    @property
    def longitudes(self):
        return self.cache["longitude"]

    @property
    def latitudes(self):
        return self.cache["latitude"]

    def times(self, path):
        key = (path, "time")
        if key not in self.cache:
            with netCDF4.Dataset(path) as dataset:
                var = dataset.variables["time"]
                values = netCDF4.num2date(var[:], units=var.units)
            self.cache[key] = values
        return self.cache[key]

    def image(self, valid_time):
        path = self.find_path(valid_time)
        times = self.times(path)
        itime = self.nearest_index(times, valid_time)
        return self.load_image(path, itime)

    def load_image(self, path, itime):
        with netCDF4.Dataset(path) as dataset:
            values = dataset.variables["data"][itime]
        return geo.stretch_image(
                self.longitudes,
                self.latitudes,
                values)

    def find_path(self, time):
        date = self.nearest_before(self.dates, time)
        i = self.nearest_index(self.dates, date)
        return self.paths[i]

    def nearest_before(self, times, time):
        if isinstance(times, list):
            times = np.asarray(times)
        diffs = self.total_seconds(times - time)
        pts = diffs <= 0
        before_times = times[pts]
        i = np.argmin(np.abs(diffs[pts]))
        return before_times[i]

    def nearest_index(self, times, time):
        if isinstance(times, list):
            times = np.asarray(times)
        seconds = self.total_seconds(times - time)
        return np.argmin(np.abs(seconds))
