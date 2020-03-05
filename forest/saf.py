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
from scipy.interpolate import griddata, LinearNDInterpolator
#from metpy.interpolate import interpolate_to_grid
from scipy.spatial import Delaunay
import itertools

import netCDF4

from forest.gridded_forecast import _to_datetime, empty_image, coordinates
from forest.util import timeout_cache

from forest import geo

#from functools import lru_cache

class saf(object):
    tri = None

    def __init__(self, pattern, label=None, locator=None):
        '''Object to process SAF NetCDF files

        :pattern: shell-style glob pattern of input file(s)'''

        if(locator):
            self.locator = locator
        else:
            self.locator = Locator(pattern)        

        if(label):
            self.label = label

    #@lru_cache(maxsize=16)
    def image(self, state):
        '''gets actual data. 

        `values` passed to `geo.stretch_image` must be a NumPy Masked Array, 
        rather than a NetCDF4 Variable, so need to add `[:]`.

        X and Y passed to :meth:`geo.stretch_image` must be 1D arrays. NWCSAF data 
        are not on a regular grid so must be regridded.

        `values` passed to :meth:`geo.stretch_image` must be a NumPy Masked Array. 

        :param state: Bokeh State object of info from UI
        :returns: Output data from :meth:`geo.stretch_image`'''
        data = empty_image()
        for nc in self.locator._sets: 
            if str(datetime.datetime.strptime(nc.nominal_product_time.replace('Z','UTC'), '%Y-%m-%dT%H:%M:%S%Z')) == state.valid_time and self.locator.varlist[state.variable] in nc.variables:
                #regrid to regular grid
                x = nc['lon'][:].flatten() # lat & lon both 2D arrays
                y = nc['lat'][:].flatten() #
                z = nc[self.locator.varlist[state.variable]][:].flatten()

                #define regular grid
                xi, yi = np.meshgrid(
                        np.linspace(x.min(),x.max(),nc.dimensions['nx'].size),
                        np.linspace(y.min(),y.max(),nc.dimensions['ny'].size), 
                            )

                #define Delaunay Triangulation
                #https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.spatial.Delaunay.html
                #does it once and re-uses for speed
                if not self.tri: 
                    self.tri = Delaunay(np.array([x,y]).transpose())

                #Interpolate onto regular grid
                interpolator = LinearNDInterpolator(self.tri, z)

                zi = interpolator(np.array([xi,yi]).transpose())

                zi = np.ma.masked_invalid(zi, copy=False)
                zi = np.ma.masked_outside(zi, nc[self.locator.varlist[state.variable]].valid_range[0], nc[self.locator.varlist[state.variable]].valid_range[1], copy=False)
                data = geo.stretch_image(xi[0,:], yi[:,0], zi)
                #data = geo.stretch_image(x[0,:], y[:,0], nc[state.variable][:])
                data.update(coordinates(state.valid_time, state.initial_time, state.pressures, state.pressure))
                data.update({
                    'name': [str(nc[self.locator.varlist[state.variable]].long_name)],
                })
                if 'units' in nc[self.locator.varlist[state.variable]].ncattrs():
                    data.update({
                        'units': [str(nc[self.locator.varlist[state.variable]].units)]
                    })

          
        return data
          
class Locator(object):
    def __init__(self, pattern):
        self.pattern = pattern
        self._sets = []
        for path in self.paths:
            #possibly use MFDataset which takes a glob pattern
            self._sets.append(netCDF4.Dataset(path)) 

        #Get variable names and keys
        self.varlist = {}
        for nc in self._sets: 
            for variable in nc.variables:
                #only display vars with lon/lat coords
                if('coordinates' in nc.variables[variable].ncattrs() and nc.variables[variable].coordinates == "lon lat"):
                    self.varlist[nc.variables[variable].long_name] = variable


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

        #return list of vars from Locator
        return self.locator.varlist.keys()

    def valid_times(self, pattern, variable):
        '''Gets valid times from input files

        :param pattern: Glob of file paths
        :param variable: String of variable name
        :return: List of Date strings
        '''
        self.locator = Locator(pattern)
        times = []
        for nc in self.locator._sets:
            if variable is None or self.locator.varlist[variable] in nc.variables:
                times.append(str(datetime.datetime.strptime(nc.nominal_product_time.replace('Z','UTC'), '%Y-%m-%dT%H:%M:%S%Z')))
        return times

    def pressures(self, path, variable):
        '''There's no pressure levels in SAF data.
        
        :returns: Nothing
        '''
        return 
