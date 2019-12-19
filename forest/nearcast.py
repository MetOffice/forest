"""
NearCast
--------------------------------------
"""
import os
import glob
import re
import datetime as dt
import pygrib as pg
import numpy as np
from forest import geo
from forest.util import timeout_cache
from forest.exceptions import FileNotFound

import inspect

NEARCAST_TOOLTIPS = [("Name", "@name"),
                     ("Value", "@image @units"),
                     ('Valid', '@valid'),
                     ("Sigma Layer", "@layer")]

class NearCast(object):
    def __init__(self, pattern):
        self.locator = Locator(pattern)

    def image(self, state):    
        imageData = self.get_grib2_data(state.pattern, state.valid_time, state.variable, state.pressure)
        
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
        
class Locator(object):
    def __init__(self, pattern):
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
        initTime = Locator.parse_date(path)
        return initTime

    def variables(self, path):
        messages = pg.open(path)
        varList = []
        for message in messages.select():
            varList.append(message['name'])
        messages.close()
        return list(set(varList))

    def valid_times(self, path, variable):
        messages = pg.index(path, "name")
        validTimeList = []
        for message in messages.select(name=variable):
            validTime = "{0:8d}{1:04d}".format(message["validityDate"], message["validityTime"])
            validTimeList.append(dt.datetime.strptime(validTime, "%Y%m%d%H%M"))
        messages.close()
        return list(set(validTimeList))

    def pressures(self, path, variable):
        messages = pg.index(path, "name")
        pressureList = []
        for message in messages.select(name=variable):
            pressureList.append(message["scaledValueOfFirstFixedSurface"])
        messages.close()
        return list(set(pressureList))
