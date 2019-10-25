import datetime
import collections
import glob
import re
import os

import numpy as np
import netCDF4

from forest.gridded_forecast import _to_datetime, empty_image
from forest.util import timeout_cache

from forest import geo

class saf(object):
    def __init__(self, pattern):
        '''Object to process SAF NetCDF files

        :pattern: shell-style glob pattern of input file(s)'''
        self.locator = Locator(pattern)        

    def image(self, state):
        '''gets actual data

        :state: object of info from UI'''
        for nc in self.locator._sets[0:1]: #just do one for now
            if nc is None:
                data = empty_image()
            else:
                data = geo.stretch_image(nc.variables['lon'][:][0], nc.variables['lat'][:][:,0], nc.variables[state.variable][:])
                return data
          
class Locator(object):
    def __init__(self, pattern):
        print("saf.Locator('{}')".format(pattern))
        self.pattern = pattern
        self._sets = []
        for path in self.find(pattern):
            #possibly use MFDataset which takes a glob pattern
            self._sets.append(netCDF4.Dataset(path)) 

    def find_file(self, valid_date):
        paths = np.array(self.paths)  # Note: timeout cache in use
        if len(paths) > 0:
            return paths[0]
        else:
            raise FileNotFound("SAF: '{}' not found".format(valid_date))

    @property
    def paths(self):
        return self.find(self.pattern)

    @staticmethod
    @timeout_cache(datetime.timedelta(minutes=10))
    def find(pattern):
        return sorted(glob.glob(pattern))

    def dates(self, paths):
        return np.array([
            self.parse_date(p) for p in paths],
            dtype='datetime64[s]')

    @staticmethod
    def parse_date(path):
        '''Parses a date from a pathname

        :path: string representation of a path
        :returns: python Datetime object
        '''
        # filename of form S_NWC_CTTH_MSG4_GuineaCoast-VISIR_20191021T134500Z.nc 
        groups = re.search("[0-9]{8}T[0-9]{6}Z", os.path.basename(path))
        if groups is not None:
            return datetime.datetime.strptime(groups[0].replace('Z','UTC'), "%Y%m%dT%H%M%S%Z") # always UTC

class Coordinates(object):
    """Menu system interface"""
    def initial_time(self, path):
        times = self.valid_times(path, None)
        if len(times) > 0:
            return times[0]
        return None

    def variables(self, path):
        '''
        Get list of variables.

         :return: list of strings of variable names'''
        nc = netCDF4.Dataset(path) 
        return nc.variables

    def valid_times(self, path, variable):
        date = Locator.parse_date(path)
        if date is None:
            return []
        return [str(date)]

    def pressures(self, path, variable):
        '''There's no pressure levels in SAF data'''
        return 
