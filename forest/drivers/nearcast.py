"""
NearCast
--------------------------------------
"""
import os
import glob
import re
import datetime as dt
import numpy as np
import forest.map_view
from forest import geo
from forest.util import timeout_cache
from forest.drivers.gridded_forecast import _to_datetime

try:
    import pygrib as pg
except ModuleNotFoundError:
    pg = None

NEARCAST_TOOLTIPS = [("Name", "@name"),
                     ("Value", "@image @units"),
                     ('Valid', '@valid'),
                     ("Sigma Layer", "@layer")]


class Dataset:
    def __init__(self, pattern=None, **kwargs):
        self.pattern = pattern
        self.loader = NearCast(self.pattern)

    def navigator(self):
        return Navigator(self.pattern)

    def map_view(self, color_mapper):
        return forest.map_view.map_view(self.loader,
                                        color_mapper,
                                        tooltips=NEARCAST_TOOLTIPS)


class NearCast(object):
    """View responsible for plotting Nearcast dataset"""
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
            imageData = self.get_grib2_data(paths[0],
                                            state.valid_time,
                                            state.variable,
                                            state.pressure)
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
        return geo.stretch_image(imageData["longitude"],
                                 imageData["latitude"],
                                 imageData["data"])

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
    """Simplified navigator
    """
    def __init__(self, pattern):
        self.pattern = pattern
        self.locator = Locator(pattern)

    def variables(self, pattern):
        """Names of variables in dataset"""
        paths = self.locator.find(self.pattern)
        if len(paths) == 0:
            return []
        return list(sorted(set(self._variables(paths[-1]))))

    @staticmethod
    def _variables(path):
        messages = pg.open(path)
        for message in messages.select():
            yield message['name']
        messages.close()

    def initial_times(self, pattern, variable=None):
        """Model initialisation times"""
        paths = self.locator.find(self.pattern)
        return list(sorted(set(Locator.parse_date(path) for path in paths)))

    def valid_times(self, pattern, variable, initial_time):
        """Validity times"""
        return self._dim(self._valid_times, variable, initial_time)

    @staticmethod
    def _valid_times(variable, path):
        messages = pg.index(path, "name")
        try:
            for message in messages.select(name=variable):
                validTime = "{0:8d}{1:04d}".format(message["validityDate"],
                                                   message["validityTime"])
                yield dt.datetime.strptime(validTime, "%Y%m%d%H%M")
        except ValueError:
            # messages.select(name=variable) raises ValueError if not found
            pass
        messages.close()

    def pressures(self, pattern, variable, initial_time):
        """Vertical coordinate"""
        return self._dim(self._pressures, variable, initial_time)

    @staticmethod
    def _pressures(variable, path):
        messages = pg.index(path, "name")
        try:
            for message in messages.select(name=variable):
                yield message["scaledValueOfFirstFixedSurface"]
        except ValueError:
            # messages.select(name=variable) raises ValueError if not found
            pass
        messages.close()

    def _dim(self, method, variable, initial_time):
        paths = self.locator.find_paths(initial_time)
        values = []
        for path in paths:
            for value in method(variable, path):
                values.append(value)
        return list(sorted(set(values)))


class Locator(object):
    """Locate files on disk"""
    def __init__(self, pattern):
        self.pattern = pattern
        self._initial_time_to_path = {}

    def find_paths(self, initial_time):
        """Find paths by initial time"""
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
