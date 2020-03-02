"""
NearCast
--------------------------------------
"""
import os
import glob
import re
import datetime as dt
import numpy as np
from forest import geo
from forest.util import timeout_cache
from forest.exceptions import FileNotFound
from forest.gridded_forecast import _to_datetime

try:
    import pygrib as pg
except ModuleNotFoundError:
    pg = None

NEARCAST_TOOLTIPS = [("Name", "@name"),
                     ("Value", "@image @units"),
                     ('Valid', '@valid'),
                     ("Sigma Layer", "@layer")]

class NearCast(object):
    def __init__(self, pattern):
        self.locator = Locator(pattern)
        self.empty_image = {
            "x": [],
            "y": [],
            "dw": [],
            "dh": [],
            "image": [],
            "name": [],
            "units": [],
            "valid": [],
            "layer": [],
        }

    def image(self, state):
        paths = self.locator.find_paths(state.initial_time)
        if len(paths) == 0:
            return self.empty_image

        try:
            imageData = self.get_grib2_data(paths[0], state.valid_time, state.variable, state.pressure)
        except ValueError:
            # TODO: Fix this properly
            return self.empty_image

        data = self.load_image(imageData)
        data.update({"name" : [imageData["name"]],
                     "units" : [imageData["units"]],
                     "valid" : [imageData["valid"]],
                     "layer" : [imageData["layer"]]})
        return data

    def load_image(self, imageData):
        return geo.stretch_image(
                imageData["longitude"], imageData["latitude"], imageData["data"])

    def get_grib2_data(self, path, valid_time, variable, pressure):
        cache = {}

        validTime = dt.datetime.strptime(str(valid_time), "%Y-%m-%d %H:%M:%S")
        vTime = "{0:d}{1:02d}".format(validTime.hour, validTime.minute)

        messages = pg.index(path, "name", "scaledValueOfFirstFixedSurface", "validityTime")
        if len(path) > 0:
            field = messages.select(name=variable, scaledValueOfFirstFixedSurface=int(pressure), validityTime=vTime)[0]
            cache["longitude"] = field.latlons()[1][0,:]
            cache["latitude"] = field.latlons()[0][:,0]
            cache["data"] = field.values
            cache["units"] = field.units
            cache["name"] = field.name
            cache["valid"] = "{0:02d}:{1:02d} UTC".format(validTime.hour, validTime.minute)
            cache["initial"] = "blah"
            scaledLowerLevel = float(field.scaledValueOfFirstFixedSurface)
            scaleFactorLowerLevel = float(field.scaleFactorOfFirstFixedSurface)
            lowerSigmaLevel = str(round(scaledLowerLevel * 10**-scaleFactorLowerLevel, 2))
            scaledUpperLevel = float(field.scaledValueOfSecondFixedSurface)
            scaleFactorUpperLevel = float(field.scaleFactorOfSecondFixedSurface)
            upperSigmaLevel = str(round(scaledUpperLevel * 10**-scaleFactorUpperLevel, 2))
            cache['layer'] = lowerSigmaLevel+"-"+upperSigmaLevel
        messages.close()
        return cache


class Navigator:
    """Simplified navigator"""
    def __init__(self, pattern):
        self.pattern = pattern
        self.locator = Locator(pattern)

    def variables(self, pattern):
        paths = self.locator.find(self.pattern)
        if len(paths) == 0:
            return []
        return list(sorted(Coordinates.variables(paths[-1])))

    def initial_times(self, pattern, variable=None):
        paths = self.locator.find(self.pattern)
        return list(sorted(set([Locator.parse_date(path) for path in paths])))

    def valid_times(self, pattern, variable, initial_time):
        return self._dim(Coordinates.valid_times, variable, initial_time)

    def pressures(self, pattern, variable, initial_time):
        return self._dim(Coordinates.pressures, variable, initial_time)

    def _dim(self, method, variable, initial_time):
        paths = self.locator.find_paths(initial_time)
        def wrapped(path):
            return method(path, variable)
        return self._collect(wrapped, paths)

    def _collect(self, method, args):
        values = []
        for arg in args:
            values += method(arg)
        return list(sorted(set(values)))


class Locator(object):
    def __init__(self, pattern):
        self.pattern = pattern
        self._initial_time_to_path = {}

    def find_paths(self, initial_time):
        self.sync()
        key = str(_to_datetime(initial_time))
        try:
            return [self._initial_time_to_path[key]]
        except KeyError:
            return []

    def sync(self):
        paths = self.find(self.pattern)
        for path in paths:
            key = str(self.parse_date(path))
            self._initial_time_to_path[key] = path

    @staticmethod
    @timeout_cache(dt.timedelta(minutes=10))
    def find(pattern):
        return sorted(glob.glob(pattern))

    @staticmethod
    def parse_date(path):
        groups = re.search("[0-9]{8}_[0-9]{4}", os.path.basename(path))
        if groups is not None:
            return dt.datetime.strptime(groups[0], "%Y%m%d_%H%M")


class Coordinates(object):
    """Menu system interface"""
    @staticmethod
    def variables(path):
        messages = pg.open(path)
        varList = []
        for message in messages.select():
            varList.append(message['name'])
        messages.close()
        return list(set(varList))

    @staticmethod
    def valid_times(path, variable):
        messages = pg.index(path, "name")
        validTimeList = []
        for message in messages.select(name=variable):
            validTime = "{0:8d}{1:04d}".format(message["validityDate"], message["validityTime"])
            validTimeList.append(dt.datetime.strptime(validTime, "%Y%m%d%H%M"))
        messages.close()
        return list(set(validTimeList))

    @staticmethod
    def pressures(path, variable):
        messages = pg.index(path, "name")
        pressureList = []
        for message in messages.select(name=variable):
            pressureList.append(message["scaledValueOfFirstFixedSurface"])
        messages.close()
        return list(set(pressureList))
