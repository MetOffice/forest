"""
SAF Driver
----------

Loads data from NWCSAF satellite NetCDF files.

.. autoclass:: Loader
    :members:

.. autoclass:: Locator
    :members:

.. autoclass:: Navigator
    :members:

"""
import datetime as dt
import glob
import re
import os
import netCDF4
from forest.gridded_forecast import empty_image, coordinates
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


class Loader:
    def __init__(self, locator, label=None):
        '''Object to process SAF NetCDF files'''
        self.locator = locator
        self.label = label

    @lru_cache(maxsize=16)
    def image(self, state):
        '''Gets actual data.

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
        self.locator.sync()
        for path in self.locator.find_paths(valid_time):
            with netCDF4.Dataset(path) as nc:
                x = nc['lon'][:]
                y = nc['lat'][:]
                var = nc[self.locator.long_name_to_variable[variable]]
                z = var[:]
                data = geo.stretch_image(x, y, z)
                data.update(coordinates(valid_time, initial_time, pressures, pressure))
                data['name'] = [str(var.long_name)]
                if 'units' in var.ncattrs():
                    data['units'] = [str(var.units)]
        return data


class Locator:
    """Locate SAF files"""
    def __init__(self, pattern):
        self.pattern = pattern
        self._paths = []
        self._date_to_path = {}
        self.long_name_to_variable = {}

    def find_paths(self, date):
        """Find a file(s) containing information related to date"""
        if date in self._date_to_path:
            return [self._date_to_path[date]]
        else:
            return []

    def variables(self):
        """Available variables"""
        return list(sorted(self.long_name_to_variable.keys()))

    def valid_times(self):
        """Available validity times"""
        return list(sorted(self._date_to_path.keys()))

    @timeout_cache(dt.timedelta(minutes=10))
    def sync(self):
        """Synchronize with file system"""
        self._paths = self._find(self.pattern)
        self._date_to_path = {self._parse_date(path): path for path in self._paths}
        if len(self.long_name_to_variable) == 0:
            # Get variable names and keys (performed once)
            for path in self._paths[-1:]:
                with netCDF4.Dataset(path) as nc:
                    for variable, var in nc.variables.items():
                        # Only display variables with lon/lat coords
                        if('coordinates' in var.ncattrs() and var.coordinates == "lon lat"):
                            self.long_name_to_variable[var.long_name] = variable

    @staticmethod
    def _find(pattern):
        return sorted(glob.glob(pattern))

    @staticmethod
    def _parse_date(path):
        '''Parses a date from a pathname

        :param path: string representation of a path
        :returns: python Datetime object
        '''
        # filename of form S_NWC_CTTH_MSG4_GuineaCoast-VISIR_20191021T134500Z.nc
        groups = re.search("[0-9]{8}T[0-9]{6}Z", os.path.basename(path))
        if groups is not None:
            return dt.datetime.strptime(groups[0].replace('Z','UTC'), "%Y%m%dT%H%M%S%Z") # always UTC


class Navigator:
    """Menu system interface

    .. note:: This is a facade or adapter for the navigator interface
    """
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
        self.locator.sync()
        return self.locator.variables()

    def valid_times(self, pattern, variable, initial_time):
        '''Gets valid times from input files

        :param pattern: Glob of file paths
        :param variable: String of variable name
        :return: List of Date strings
        '''
        self.locator.sync()
        return self.locator.valid_times()

    def pressures(self, path, variable, initial_time):
        '''There's no pressure levels in SAF data.

        :returns: empty list
        '''
        return []
