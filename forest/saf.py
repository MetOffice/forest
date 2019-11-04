"""
SAF Loader
----------

Loads data from NWCSAF satellite NetCDF files.

.. autoclass:: saf
    :members:

.. autoclass:: Locator
    :members:

.. autoclass:: Coordinates
    :members:

"""

import datetime
import collections
import glob
import re
import os

import numpy as np
import numpy.ma as ma
from scipy.interpolate import griddata
import netCDF4

from forest.gridded_forecast import _to_datetime, empty_image
from forest.util import timeout_cache

from forest import geo

class saf(object):
    def __init__(self, pattern, label=None, locator=None):
        '''Object to process SAF NetCDF files

        :pattern: shell-style glob pattern of input file(s)'''
        if(locator):
            self.locator = locator
        else:
            self.locator = Locator(pattern)        

        if(label):
            self.label = label

    def image(self, state):
        '''gets actual data. 

        X and Y passed to :meth:`geo.stretch_image` must be 1D arrays. NWCSAF data 
        are not on a regular grid so must be regridded.

        `values` passed to :meth:`geo.stretch_image` must be a NumPy Masked Array. 

        :param state: Bokeh State object of info from UI
        :returns: Output data from :meth:`geo.stretch_image`'''
        data = empty_image()
        print("wibble", "saf.image called")
        for nc in self.locator._sets: 
            if str(datetime.datetime.strptime(nc.nominal_product_time.replace('Z','UTC'), '%Y-%m-%dT%H:%M:%S%Z')) == state.valid_time and state.variable in nc.variables:
                #regrid to regular grid
                x = nc['lon'][:] # lat & lon both 2D arrays
                y = nc['lat'][:] #

                #define grid
                xi, yi = np.meshgrid(
                        np.linspace(x.min(),x.max(),nc.dimensions['nx'].size),
                        np.linspace(y.min(),y.max(),nc.dimensions['ny'].size), 
                            )

                zi = griddata(np.array([x.flatten(),y.flatten()]).transpose(),nc[state.variable][:].flatten(), (xi, yi))

                data = geo.stretch_image(xi[0,:], yi[:,0], np.nan_to_num(zi))
          
        return data
          
class Locator(object):
    def __init__(self, pattern):
        self.pattern = pattern
        self._sets = []
        for path in self.paths:
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

        :param path: string representation of a path
        :returns: python Datetime object
        '''
        # filename of form S_NWC_CTTH_MSG4_GuineaCoast-VISIR_20191021T134500Z.nc 
        groups = re.search("[0-9]{8}T[0-9]{6}Z", os.path.basename(path))
        if groups is not None:
            return datetime.datetime.strptime(groups[0].replace('Z','UTC'), "%Y%m%dT%H%M%S%Z") # always UTC

class Coordinates(object):
    """Menu system interface"""
    def initial_time(self, pattern):
        '''Return initial time.

        :param pattern: Glob pattern of filepaths
        :returns: Python Datetime object
        '''
        times = self.valid_times(pattern, None)
        if len(times) > 0:
            return times[0]
        return None

    def variables(self, pattern):
        '''Get list of variables.

         :param pattern: glob pattern of filepaths
         :returns: list of strings of variable names
         '''
        self.locator = Locator(pattern)        
        varlist  = []
        for nc in self.locator._sets: 
            varlist = varlist + list(nc.variables.keys())
    
        #return list of vars. coercing to set ensures uniqueness
        return list(set(varlist))

    def valid_times(self, pattern, variable):
        '''Gets valid times from input files

        :param pattern: Glob of file paths
        :param variable: String of variable name
        :return: List of Date strings
        '''
        self.locator = Locator(pattern)
        times = []
        for nc in self.locator._sets:
            if variable is None or variable in nc.variables:
                times.append(str(datetime.datetime.strptime(nc.nominal_product_time.replace('Z','UTC'), '%Y-%m-%dT%H:%M:%S%Z')))
        return times

    def pressures(self, path, variable):
        '''There's no pressure levels in SAF data.
        
        :returns: Nothing
        '''
        return 
