"""
NearCast
--------------------------------------
"""
import os
import glob
import re
import datetime as dt
#import pygrib as pg
import numpy as np
from forest import geo
from forest.util import timeout_cache
from forest.exceptions import FileNotFound

class Nearcast(object):
    def __init__(self, pattern):
        self.locator = Locator(pattern)        

    def image(self, date):
        file_name = self.locator.find_file(date)
        return self.load_image(file_name)

    def load_image(self, path):
        imageData = self.get_data(path)
        return geo.stretch_image(
                imageData["longitude"], imageData["latitude"], imageData["data"])

    def get_data(self, path):
        self.cache = {}
        gribFields = pg.index(path, "name", "scaledValueOfFirstFixedSurface", "forecastTime")
        if len(path) > 0:
            field = gribFields.select(name="Precipitable water", scaledValueOfFirstFixedSurface=699999988, forecastTime=0)[0]
            
            self.cache["longitude"] = field.latlons()[1][0,:]
            self.cache["latitude"] = field.latlons()[0][:,0]
            self.cache["data"] = field.values
        gribFields.close()
        return self.cache

class Locator(object):
    def __init__(self, pattern):
        print("nearcast.Locator('{}')".format(pattern))
        self.pattern = pattern

    def find_file(self, valid_date):
        paths = np.array(self.paths)  # Note: timeout cache in use
        if len(paths) > 0:
            return paths[0]
        else:
            raise FileNotFound("NearCast: '{}' not found".format(valid_date))

    @property
    def paths(self):
        return self.find(self.pattern)

    @staticmethod
    @timeout_cache(dt.timedelta(minutes=10))
    def find(pattern):
        return sorted(glob.glob(pattern))

    def dates(self, paths):
        return np.array([
            self.parse_date(p) for p in paths],
            dtype='datetime64[s]')

    @staticmethod
    def parse_date(path):
        groups = re.search("[0-9]{8}_[0-9]{4}", os.path.basename(path))
        if groups is not None:
            return dt.datetime.strptime(groups[0], "%Y%m%d_%H%M")


class Coordinates(object):
    """Menu system interface"""
    def initial_time(self, path):
        times = self.valid_times(path, None)
        if len(times) > 0:
            return times[0]
        return None

    def variables(self, path):
        return ["NearCast"]

    def valid_times(self, path, variable):
        #gribMessages = pg.
        date = Locator.parse_date(path)
        if date is None:
            return []
        return [str(date)]

    def pressures(self, path, variable):
        return 
