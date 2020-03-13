"""
SAF Loader
----------

Loads data from NWCSAF satellite NetCDF files.

.. autoclass:: saf
    :members:

.. autoclass:: Locator
    :members:

.. autoclass:: Navigator
    :members:

"""

import datetime as dt
import collections
import glob
import re
import os

import numpy as np
import numpy.ma as ma
from scipy.interpolate import griddata
#from metpy.interpolate import interpolate_to_grid
import netCDF4

from forest.gridded_forecast import _to_datetime, empty_image, coordinates
from forest.util import timeout_cache

from forest import geo, view

from functools import lru_cache


class Dataset:
    """High-level NWC/SAF dataset"""
    def __init__(self,
                 label=None,
                 pattern=None,
                 color_mapper=None):
        self.label = label
        self.pattern = pattern
        self.color_mapper = color_mapper
        self.locator = Locator(self.pattern)

    def navigator(self):
        """Construct navigator"""
        return Navigator()

    def map_view(self):
        """Construct view"""
        loader = saf(self.pattern, self.label, self.locator)
        return view.UMView(loader, self.color_mapper)


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

    @lru_cache(maxsize=16)
    def image(self, state):
        '''gets actual data.

        X and Y passed to :meth:`geo.stretch_image` must be 1D arrays. NWCSAF data
        are not on a regular grid so must be regridded.

        `values` passed to :meth:`geo.stretch_image` must be a NumPy Masked Array.

        :param state: Bokeh State object of info from UI
        :returns: Output data from :meth:`geo.stretch_image`'''
        return self._image(state.variable,
                           state.initial_time,
                           state.valid_time,
                           state.pressures,
                           state.pressure)

    def _image(self, variable, initial_time, valid_time, pressures, pressure):
        data = empty_image()
        for path in self.locator.paths:
            with netCDF4.Dataset(path) as nc:
                if str(dt.datetime.strptime(nc.nominal_product_time.replace('Z','UTC'), '%Y-%m-%dT%H:%M:%S%Z')) == valid_time and self.locator.varlist[variable] in nc.variables:
                    #regrid to regular grid
                    x = nc['lon'][:].flatten() # lat & lon both 2D arrays
                    y = nc['lat'][:].flatten() #
                    z = nc[self.locator.varlist[variable]][:].flatten()

                    #define grid
                    xi, yi = np.meshgrid(
                            np.linspace(x.min(),x.max(),nc.dimensions['nx'].size),
                            np.linspace(y.min(),y.max(),nc.dimensions['ny'].size),
                                )

                    zi = griddata(
                            np.array([x,y]).transpose(),
                            z,
                            (xi, yi),
                            method='linear',
                            fill_value=np.nan)

                    zi = np.ma.masked_invalid(zi, copy=False)
                    zi = np.ma.masked_outside(zi, nc[self.locator.varlist[variable]].valid_range[0], nc[self.locator.varlist[variable]].valid_range[1], copy=False)
                    data = geo.stretch_image(xi[0,:], yi[:,0], zi)
                    #data = geo.stretch_image(x[0,:], y[:,0], nc[variable][:])
                    data.update(coordinates(valid_time, initial_time, pressures, pressure))
                    data.update({
                        'name': [str(nc[self.locator.varlist[variable]].long_name)],
                    })
                    if 'units' in nc[self.locator.varlist[variable]].ncattrs():
                        data.update({
                            'units': [str(nc[self.locator.varlist[variable]].units)]
                        })
        return data


class Locator(object):
    def __init__(self, pattern):
        self.pattern = pattern

        # Get variable names and keys
        self.long_name_to_variable = {}
        for path in self.paths:
            with netCDF4.Dataset(path) as nc:
                for variable, var in nc.variables.items():
                    # Only display variables with lon/lat coords
                    if('coordinates' in var.ncattrs() and var.coordinates == "lon lat"):
                        self.long_name_to_variable[var.long_name] = variable

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
        '''Parses a date from a pathname

        :param path: string representation of a path
        :returns: python Datetime object
        '''
        # filename of form S_NWC_CTTH_MSG4_GuineaCoast-VISIR_20191021T134500Z.nc
        groups = re.search("[0-9]{8}T[0-9]{6}Z", os.path.basename(path))
        if groups is not None:
            return dt.datetime.strptime(groups[0].replace('Z','UTC'), "%Y%m%dT%H%M%S%Z") # always UTC


class Navigator:
    """Menu system interface"""
    def initial_times(self, pattern, variable):
        """Satellite data has no concept of initial time"""
        return [dt.datetime(1970, 1, 1)]

    def variables(self, pattern):
        '''Get list of variables.

         :param pattern: glob pattern of filepaths
         :returns: list of strings of variable names
         '''
        self.locator = Locator(pattern)
        return list(sorted(self.locator.long_name_to_variable.keys()))

    def valid_times(self, pattern, variable, initial_time):
        '''Gets valid times from input files

        :param pattern: Glob of file paths
        :param variable: String of variable name
        :return: List of Date strings
        '''
        return [dt.datetime(1970, 1, 1)] # For now

        self.locator = Locator(pattern)
        times = []
        for nc in self.locator._sets:
            if variable is None or self.locator.varlist[variable] in nc.variables:
                times.append(str(dt.datetime.strptime(nc.nominal_product_time.replace('Z','UTC'), '%Y-%m-%dT%H:%M:%S%Z')))
        return times

    def pressures(self, path, variable, initial_time):
        '''There's no pressure levels in SAF data.

        :returns: empty list
        '''
        return []
