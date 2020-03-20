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
from functools import partial
import datetime as dt
import glob
import re
import os
import netCDF4
from forest.drivers.gridded_forecast import empty_image, coordinates
import forest.util
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

    def _image(self, long_name, initial_time, valid_time, pressures, pressure):
        data = empty_image()
        paths = self.locator.glob()
        long_name_to_variable = self.locator.long_name_to_variable(paths)
        for path in self.locator.find_paths(paths, valid_time):
            with netCDF4.Dataset(path) as nc:
                if long_name not in long_name_to_variable:
                    continue
                x = nc['lon'][:]
                y = nc['lat'][:]
                var = nc[long_name_to_variable[long_name]]
                z = var[:]
                data = geo.stretch_image(x, y, z)
                data.update(coordinates(valid_time, initial_time, pressures, pressure))
                data['name'] = [str(var.long_name)]
                if 'units' in var.ncattrs():
                    data['units'] = [str(var.units)]
        return data


@forest.util.timeout_cache(dt.timedelta(minutes=10))
def cached_glob(pattern):
    """Glob file system at most once every 10 minutes for a pattern"""
    return sorted(glob.glob(pattern))


class FileNameLocator:
    """Find files with date information encoded in name"""
    def __init__(self, regex, fmt):
        self.parse_date = partial(forest.util.parse_date, regex, fmt)

    def find_paths(self, paths, date):
        """Find a file(s) by date"""
        for path in paths:
            if self.parse_date(path) == date:
                yield path


class Locator:
    """Locate SAF files"""
    def __init__(self, pattern):
        self.pattern = pattern
        self._locators = {
            "file_name": FileNameLocator("[0-9]{8}T[0-9]{6}Z", "%Y%m%dT%H%M%S%Z")
        }

    def glob(self):
        """List file system"""
        return cached_glob(self.pattern)

    def find_paths(self, paths, date):
        """Find a file(s) containing information related to date"""
        return self._locators["file_name"].find_paths(paths, date)

    def variables(self, paths):
        """Available variables"""
        return list(sorted(self.long_name_to_variable(paths).keys()))

    def valid_times(self, paths):
        """Available validity times"""
        for path in paths:
            yield self._locators["file_name"].parse_date(path)

    def long_name_to_variable(self, paths):
        """Map long_name attrs to variables"""
        mapping = {}
        for path in paths[-1:]:
            mapping.update(self._read_long_name_to_variable(path))
        return mapping

    @staticmethod
    @lru_cache(maxsize=1)
    def _read_long_name_to_variable(path):
        mapping = {}
        with netCDF4.Dataset(path) as nc:
            for variable, var in nc.variables.items():
                # Only display variables with lon/lat coords
                if('coordinates' in var.ncattrs() and var.coordinates == "lon lat"):
                    mapping[var.long_name] = variable
        return mapping


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
        return self.locator.variables(self.locator.glob())

    def valid_times(self, pattern, variable, initial_time):
        '''Gets valid times from input files

        :param pattern: Glob of file paths
        :param variable: String of variable name
        :return: List of Date strings
        '''
        return list(sorted(self.locator.valid_times(self.locator.glob())))

    def pressures(self, path, variable, initial_time):
        '''There's no pressure levels in SAF data.

        :returns: empty list
        '''
        return []
