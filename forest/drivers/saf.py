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
        return Navigator(self.locator)

    def map_view(self):
        """Construct view"""
        loader = Loader(self.locator, self.label)
        return view.UMView(loader, self.color_mapper)


class Loader(object):
    def __init__(self, locator, label=None):
        '''Object to process SAF NetCDF files'''
        self.locator = locator

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
        print(valid_time, self.locator.find_paths(valid_time))
        for path in self.locator.find_paths(valid_time):
            with netCDF4.Dataset(path) as nc:
                # Regrid to regular grid
                x = nc['lon'][:].flatten() # lat & lon both 2D arrays
                y = nc['lat'][:].flatten() #
                var = nc[self.locator.long_name_to_variable[variable]]
                z = var[:].flatten()

                # TODO: Replace with datashader pipeline
                # Define grid
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
                zi = np.ma.masked_outside(zi, var.valid_range[0], var.valid_range[1], copy=False)
                data = geo.stretch_image(xi[0,:], yi[:,0], zi)
                data.update(coordinates(valid_time, initial_time, pressures, pressure))
                data['name'] = [str(var.long_name)]
                if 'units' in var.ncattrs():
                    data['units'] = [str(var.units)]
        return data


class Locator(object):
    def __init__(self, pattern):
        self.pattern = pattern

        # Parse dates
        self.date_to_path = {self.parse_date(path): path for path in self.paths}

        # Get variable names and keys
        self.long_name_to_variable = {}
        for path in self.paths[-1:]:
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

    def find_paths(self, date):
        if date in self.date_to_path:
            return [self.date_to_path[date]]
        else:
            return []

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
    def __init__(self, locator):
        self.locator = locator

    def initial_times(self, pattern, variable):
        """Satellite data has no concept of initial time"""
        return [dt.datetime(1970, 1, 1)]

    def variables(self, pattern):
        '''Get list of variables.

         :param pattern: glob pattern of filepaths
         :returns: list of strings of variable names
         '''
        return list(sorted(self.locator.long_name_to_variable.keys()))

    def valid_times(self, pattern, variable, initial_time):
        '''Gets valid times from input files

        :param pattern: Glob of file paths
        :param variable: String of variable name
        :return: List of Date strings
        '''
        return list(sorted(self.locator.date_to_path.keys()))

    def pressures(self, path, variable, initial_time):
        '''There's no pressure levels in SAF data.

        :returns: empty list
        '''
        return []
