"""Module containing a class to manage the datasets for Forest.

In particular, the ForestDataset class supports just in time loading.

Functions
---------

- get_var_lookup() -- Read config files into dictionary.
- get_model_run_times() -- Get dates of recent model runs.

Classes
-------

- ForestDataset -- Main class for containing Forest data.

"""
import os
import datetime
import datetime as dt
import configparser
import functools
import numpy
import copy
from collections import OrderedDict
from functools import lru_cache

import iris
import iris.coord_categorisation
import cf_units

import forest.util

# The number of days into the past to look for data. The current
# value specifies looking for data up to 1 week old
NUM_DATA_DAYS = 7
# The number of days in a model run. Not all models will run this long, but
# this covers the longest running models
MODEL_RUN_DAYS = 5
MODEL_OBS_COMP_DAYS = 3

MODEL_RUN_PERIOD = 12
TIME_EPSILON_HRS = 0.1
NUM_HOURS_IN_DAY = 24

COORD_WINDOW_SIZE = 0.5 # 0.5 degree window around point for interpolation for
                      # time series

WIND_SPEED_NAME = 'wind_speed'
WIND_VECTOR_NAME = 'wind_vectors'
WIND_STREAM_NAME = 'wind_streams'
WIND_MSLP_NAME = 'wind_mslp'
WIND_UNIT_MPH = 'miles-hour^-1'
WIND_UNIT_MPH_DISPLAY = 'miles per hour'

PRECIP_VAR_NAME = 'precipitation'
PRECIP_UNIT_MM = 'mm'
PRECIP_UNIT_ACCUM = 'kg-m-2'
PRECIP_UNIT_RATE = 'kg-m-2-hour^-1'

PRECIP_UNIT_ACCUM_DISPLAY = 'mm'
PRECIP_UNIT_RATE_DISPLAY = 'mm/h'

MSLP_NAME = 'mslp'
MSLP_UNIT_HPA = 'hectopascals'

VAR_NAMES = [PRECIP_VAR_NAME,
             'air_temperature',
             WIND_SPEED_NAME,
             WIND_VECTOR_NAME,
             'cloud_fraction',
             WIND_MSLP_NAME,
             'mslp',
             'x_wind',
             'y_wind',
             ]

WIND_VECTOR_VARS = ['wv_X',
                    'wv_Y',
                    'wv_U',
                    'wv_V',
                    'wv_X_grid',
                    'wv_Y_grid',
                    'wv_mag',
                    'wv_angle',
                    ]

WIND_VARS = WIND_VECTOR_VARS + [WIND_SPEED_NAME,
                                WIND_STREAM_NAME,
                                WIND_MSLP_NAME,
                                WIND_VECTOR_NAME,
                                ]

PRECIP_ACCUM_WINDOW_SIZES_LIST = [3,6,12,24]

WIND_GRID_SIZE = (40,30)

PRECIP_ACCUM_WINDOW_SIZES_DICT = dict([('accum_precip_{0}hr'.format(window1), window1) for window1 in PRECIP_ACCUM_WINDOW_SIZES_LIST])
PRECIP_ACCUM_VARS = list(PRECIP_ACCUM_WINDOW_SIZES_DICT.keys())

UNIT_DICT = {PRECIP_VAR_NAME: PRECIP_UNIT_RATE,
             'cloud_fraction': None,
             'air_temperature': 'celsius',
             'x_wind': WIND_UNIT_MPH,
             'y_wind': WIND_UNIT_MPH,
             WIND_SPEED_NAME: WIND_UNIT_MPH,
             MSLP_NAME : MSLP_UNIT_HPA,
             WIND_VECTOR_NAME: WIND_UNIT_MPH,
             WIND_STREAM_NAME: WIND_UNIT_MPH,
             WIND_MSLP_NAME: WIND_UNIT_MPH,
             }

UNIT_DICT.update(dict([(var1,WIND_UNIT_MPH) for var1 in WIND_VECTOR_VARS]))
UNIT_DICT.update(dict([(var1,PRECIP_UNIT_ACCUM) for var1 in PRECIP_ACCUM_VARS]))

UNIT_DICT_DISPLAY = {PRECIP_VAR_NAME: PRECIP_UNIT_RATE_DISPLAY,
                     'cloud_fraction': None,
                     'air_temperature': 'celsius',
                     'x_wind': WIND_UNIT_MPH_DISPLAY,
                     'y_wind': WIND_UNIT_MPH_DISPLAY,
                     WIND_SPEED_NAME: WIND_UNIT_MPH_DISPLAY,
                     'mslp': 'hectopascals',
                     WIND_VECTOR_NAME: WIND_UNIT_MPH_DISPLAY,
                     WIND_STREAM_NAME: WIND_UNIT_MPH_DISPLAY,
                     }
UNIT_DICT_DISPLAY.update(dict([(var1,WIND_UNIT_MPH_DISPLAY) for var1 in WIND_VECTOR_VARS]))
UNIT_DICT_DISPLAY.update(dict([(var1,PRECIP_UNIT_ACCUM_DISPLAY) for var1 in PRECIP_ACCUM_VARS]))



__all__ = [
    'stash_item',
    'stash_section',
    'stash_name',
    'stash_codes',
    'load_times',
    'load_cube',
    'times'
]


def stash_name(variable, convention):
    """Stash name related to variable

    :param convention: either 'ra1t' or 'ga6'
    :returns: stash code
    """
    table = stash_codes(convention)
    return table[variable]['stash_name']


def stash_item(variable, convention):
    """Stash item related to variable

    :param convention: either 'ra1t' or 'ga6'
    :returns: stash code
    """
    table = stash_codes(convention)
    return table[variable]['stash_item']


def stash_section(variable, convention):
    """Stash section related to variable

    :param convention: either 'ra1t' or 'ga6'
    :returns: stash code
    """
    table = stash_codes(convention)
    return table[variable]['stash_section']


@lru_cache(maxsize=2)
def stash_codes(convention):
    """Stash code reference related to UM systems

    :param convention: either 'ra1t' or 'ga6'
    :returns: nested dict of stash codes related to variables
    """
    path = config_file(convention.lower())
    return get_var_lookup(path)


def config_file(config):
    return os.path.join(os.path.dirname(__file__),
                        'var_list_{config}.conf'.format(config=config))


def get_var_lookup(path):
    """Read config file into dictionary

    :param path: location on disk of config file
    """
    parser = configparser.RawConfigParser()
    parser.read(path)
    field_dict = {}
    for sect1 in parser.sections():
        field_dict[sect1] = dict(parser.items(sect1))
        try:
            field_dict[sect1]['stash_section'] = \
                int(field_dict[sect1]['stash_section'])
            field_dict[sect1]['stash_item'] = \
                int(field_dict[sect1]['stash_item'])
            field_dict[sect1]['accumulate'] = \
                field_dict[sect1]['accumulate'] == 'True'
        except:
            print('warning: stash values not converted to numbers.')
    return field_dict


def load_times(cube):
    """Read datetime objects from iris cube

    :param cube: iris.Cube instance
    :returns: time axis points
    """
    coord = cube.coord('time')
    return [cell.point for cell in coord.cells()]


def times(path, section, item):
    """Read time axis from NetCDF file using stash codes

    Simple way to read the time axis related to a particular
    variable

    .. note:: The stash section and item are encoded in the
              netcdf attribute um_stash_source

    :param path: path to netcdf file
    :param section: stash code section
    :param item: stash code item
    :returns: time axis points
    """
    return load_cube(path, section, item).coord('time').points


def load_cube(path, section, item):
    """Load cube from stash codes"""
    constraint = iris.Constraint(cube_func=cube_func(section, item))
    return iris.load_cube(path, constraint)


def cube_func(section, item):
    """Create a cube function to filter stash codes"""
    def func(cube):
        return (cube.attributes['STASH'].section == section and
                cube.attributes['STASH'].item == item)
    return func


def get_available_datasets(model_run_times,
                           file_patterns,
                           file_formats,
                           file_loader):
    """
    Get a list of model runs times for which there is model output data
    available, and list of dictionaries with a dataset for each model run time.

    :param file_loader: class responsible for loading remote or local files
    :param model_run_times: times when the model was run
    """
    datasets = OrderedDict()
    for model_run_time in model_run_times:
        fct_str = format_model_run_time(model_run_time)
        fct_data_dict = {}
        model_run_data_present = True
        for name, pattern in file_patterns.items():
            file_format = file_formats[name]
            file_name = os.path.join('model_data', pattern.format(model_run_time))
            print(file_name, file_loader.file_exists(file_name))
            fct_data_dict[name] = forest.data.ForestDataset.convention(file_name,
                                                                       file_loader,
                                                                       file_format)

            model_run_data_present = model_run_data_present and file_loader.file_exists(file_name)
        # include forecast if all configs are present
        # TODO: reconsider data structure to allow for some model configs at different times to be present
        if model_run_data_present:
            datasets[fct_str] = fct_data_dict
        else:
            print("WARNING: '{}' has missing files".format(fct_str))
    return datasets


def get_model_run_times(period_start, num_days, model_run_period):
    """Create a list of model times from the last num_days days.

    Arguments
    ---------
    - num_days -- Int; Set number of days to go back and get dates for.
    - model_run_period -- Int; period of model runs in hours i.e. there
                          is a model run every model_run_period hours.
    """
    midnight = dt.datetime(period_start.year,
                           period_start.month,
                           period_start.day,
                           tzinfo=dt.timezone.utc)
    return [midnight + dt.timedelta(hours=hours)
            for hours in range(0, num_days * NUM_HOURS_IN_DAY, model_run_period)]


def format_model_run_time(time):
    return '{:%Y%m%dT%H%MZ}'.format(time)


@forest.util.timer
def get_available_times(datasets, var1):
    key0 = list(datasets.keys())[0]
    available_times = datasets[key0].get_times(var1)
    for ds_name in datasets:
        times1 = datasets[ds_name].get_times(var1)
        available_times = numpy.array([t1 for t1 in available_times if t1 in times1])
    return available_times


def check_bounds(cube1, selected_pt):
    """
    check whether the selected point falls within the bounds of the specified
    cube.
    :param cube1: The data cube with latitude and longitude metadata
    :param selected_pt: A tuple representing (latitude, longitude) coordinates.
    :return: True if the point is in the latitude and longtude range of the
             cube's metadata.
    """
    min_lat = cube1.coord('latitude').points.min()
    max_lat = cube1.coord('latitude').points.max()
    min_lon = cube1.coord('longitude').points.min()
    max_lon = cube1.coord('longitude').points.max()

    if selected_pt[0] < min_lat:
        return False
    if selected_pt[0] > max_lat:
        return False
    if selected_pt[1] < min_lon:
        return False
    if selected_pt[1] > max_lon:
        return False
    return True


def do_cube_load(
        path_to_load,
        stash_section,
        stash_item,
        time_ix,
        lat_long_coord):
    """
    Load a cube from the given path using the specified constraints. The field
    dict argument is required, but the time_ix and lat_long_coord arguments are
    optional. The cube loaded will be constrained by one or both of time and
    lat/lon if specified.
    :param path_to_load: The file path to load
    :param time_ix: A float representing the time in hours since 1970. Only
                    this time coordinate will be loaded if specified, otherwise
                    all times for the specified variable will be loaded.
    :param lat_long_coord: a lat long coordinate around which to load a half
                           degree window of points.
    :return: The iris cube of the data at the path with given constraints.
    """
    cf1 = lambda cube1: \
        cube1.attributes['STASH'].section == stash_section and \
        cube1.attributes['STASH'].item == stash_item
    coord_constraint_dict = {}

    if time_ix != ForestDataset.TIME_INDEX_ALL:
        time_obj = datetime.datetime.utcfromtimestamp(time_ix * 3600)
        time_desc = str(time_obj)
        if int(iris.__version__.split('.')[0]) == 1:
            def time_comp(time_index, eps1, cell1):
                return abs(cell1.point - time_index) < eps1

            if time_ix != ForestDataset.TIME_INDEX_ALL:
                coord_constraint_dict['time'] = \
                    functools.partial(time_comp, time_ix, 1e-5)

        elif int(iris.__version__.split('.')[0]) == 2:
            def time_comp(selected_time, eps, cell):
                return abs(cell.point - selected_time).total_seconds() < eps

            if time_ix != ForestDataset.TIME_INDEX_ALL:
                coord_constraint_dict['time'] = \
                    functools.partial(time_comp, time_obj, 1)
    else:
        time_desc = time_ix

    if lat_long_coord is not None:
        def lat_long_comp(loc1, window1, cell1):
            return abs(cell1.point - loc1) < window1

        coord_constraint_dict['latitude'] = \
            functools.partial(lat_long_comp,
                              lat_long_coord[0],
                              COORD_WINDOW_SIZE)

        coord_constraint_dict['longitude'] = \
            functools.partial(lat_long_comp,
                              lat_long_coord[1],
                              COORD_WINDOW_SIZE)
        loc_desc = \
            'loading window size {0} around point ({1},{2})'.format(
                COORD_WINDOW_SIZE,
                lat_long_coord[0],
                lat_long_coord[1],
            )
    else:
        loc_desc = 'loading all locations'

    ic1 = iris.Constraint(cube_func=cf1,
                          coord_values=coord_constraint_dict)

    print('path to load {0}'.format(path_to_load))
    print('time to load {0}'.format(time_desc))
    print(loc_desc)
    print('stash to load section {0} item {1}'.format(stash_section,
                                                      stash_item))

    try:
        dc1 = iris.load_cube(path_to_load, ic1)
    except iris.exceptions.ConstraintMismatchError:
        dc1 = None
    return dc1


class ForestDataset(object):
    """Declare main class for holding Forest data.

    Methods
    -------
    - __init__() -- Factory method.
    - __str__() -- String method.
    - get_data() -- Call self.retrieve_data() and self.load_data().
    - load_data() -- Call loader methods.
    - basic_cube_load() -- Load simple data into a cube.
    - wind_speed_loader() -- Load x/y wind data, create speed cube.
    - wind_vector_loader() -- Load x/y wind data, create vector cube.
    - add_accum_precip_keys() -- Add precip. accum. keys to data dict.
    - accum_precip_loader() -- Load precip. data and calc. accums.
    - accum_precip() -- Calculate precip. accumulations.

    Attributes
    ----------
    - var_lookup -- Dict; Links variable names to data keys.
    - file_name -- Str; Specifies netCDF file name.
    - loaders -- Dict; Dictionary of loader functions for vars.
    - data -- Dict; Loaded data cubes.
    """
    TIME_INDEX_ALL = 'all'

    @classmethod
    def convention(cls, file_name, file_loader, convention):
        """Helper to construct dataset"""
        return cls(file_name, file_loader, stash_codes(convention))

    def __init__(self,
                 file_name,
                 file_loader,
                 var_lookup):
        """ForestDataset factory function"""
        self.var_lookup = var_lookup
        self.file_name = file_name
        self.file_loader = file_loader

        # set up data loader functions
        self.cube_loaders = dict([(v1, self.basic_cube_load) for v1 in VAR_NAMES])
        self.cube_loaders[WIND_SPEED_NAME] = self.wind_speed_loader
        self.cube_loaders[WIND_STREAM_NAME] = self.wind_speed_loader

        for wv_var in WIND_VECTOR_VARS:
            self.cube_loaders[wv_var] = self.wind_vector_loader
        for accum_precip_var in PRECIP_ACCUM_VARS:
            ws1 = PRECIP_ACCUM_WINDOW_SIZES_DICT[accum_precip_var]
            self.cube_loaders[accum_precip_var] = functools.partial(self.accum_precip_loader, ws1)

        self.ts_var_names = dict([(v1, v1) for v1 in VAR_NAMES])
        self.ts_loaders = dict([(v1, self._basic_ts_load) for v1 in VAR_NAMES])
        for wv_var in WIND_VARS:
            self.ts_loaders[wv_var] = self._wind_ts_loader
        for accum_precip_var in PRECIP_ACCUM_VARS:
            ws1 = PRECIP_ACCUM_WINDOW_SIZES_DICT[accum_precip_var]
            self.ts_loaders[accum_precip_var] = \
                functools.partial(self._precip_accum_ts_load, ws1)

    def __str__(self):
        """Return string"""
        return 'FOREST dataset'

    @forest.util.timer
    def get_times(self, var_name):
        if var_name in WIND_VARS:
            return self._wind_time_load()
        elif var_name in PRECIP_ACCUM_VARS:
            return self._accum_precip_time_load(var_name)
        else:
            return self._basic_time_load(var_name)

    def _wind_time_load(self):
        return self._basic_time_load('x_wind')

    def _accum_precip_time_load(self, var_name):
        precip_times = self._basic_time_load(PRECIP_VAR_NAME)
        window_size = PRECIP_ACCUM_WINDOW_SIZES_DICT[var_name]
        return numpy.unique(numpy.floor(precip_times / window_size) * window_size) + (window_size/2.0)

    def _basic_time_load(self, var_name):
        message = "'{}' must be one of {}".format(var_name, self.var_lookup.keys())
        assert var_name in self.var_lookup, message
        path = self.file_loader.load_file(self.file_name)
        section = self.var_lookup[var_name]['stash_section']
        item = self.var_lookup[var_name]['stash_item']
        return times(path, section, item)

    @forest.util.timer
    def get_data(self, var_name, selected_time, convert_units=True):
        """Calls functions to retrieve and load data.

        Arguments
        ---------
        - var_name -- Str; Redundant: used to match other loaders.
        """
        time_ix = selected_time
        if time_ix is None:
            time_ix = ForestDataset.TIME_INDEX_ALL
        else:
            print('loading data for time {0}'.format(time_ix))

        if self.file_loader.file_exists(self.file_name):
            # Load the data into memory from file (will only load
            # metadata initially)
            cube = self.cube_loaders[var_name](var_name, time_ix)
            has_units = cube.units is not None and \
                        cube.units.name != 'unknown'
            if convert_units and has_units:
                if UNIT_DICT[var_name]:
                    cube.convert_units(UNIT_DICT[var_name])
        else:
            cube = None
        return cube

    def basic_cube_load(self, var_name, time_ix):
        """Load simple cubes.

        Arguments
        ---------
        - var_name -- Str; Var name used to define data to load.
        - time_ix -- Index of the time to load
        """
        field_dict = self.var_lookup[var_name]
        stash_item = field_dict['stash_item']
        stash_section = field_dict['stash_section']
        path_to_load = self.file_loader.load_file(self.file_name)
        return do_cube_load(path_to_load,
                            stash_section,
                            stash_item,
                            time_ix=time_ix,
                            lat_long_coord=None)

    def wind_speed_loader(self, var_name, time_ix):
        """Process wind cubes to calculate wind speed.
        Arguments
        ---------
        - var_name -- Str; Redundant: used to match other loaders.
        - time_ix -- Str; specify the time to load
        """
        print('calculating wind speed')
        cube_pow = iris.analysis.maths.exponentiate
        cube_x_wind = self.get_data('x_wind', time_ix)
        cube_y_wind = self.get_data('y_wind', time_ix)
        cube = cube_pow(cube_pow(cube_x_wind, 2.0) +
                        cube_pow(cube_y_wind, 2.0), 0.5)
        cube.rename(WIND_SPEED_NAME)
        return cube

    def wind_vector_loader(self, var_name, time_ix):
        """Gets wind data and calculates wind vectors"""
        cube_x_wind = self.get_data('x_wind', time_ix)
        cube_y_wind = self.get_data('y_wind', time_ix)
        vectors = forest.util.calc_wind_vectors(cube_x_wind,
                                                cube_y_wind,
                                                WIND_GRID_SIZE)
        return vectors[var_name]

    def accum_precip_loader(self, window_size, var_name, time_ix):
        """Gets data and creates accumulated precipitation cube.

        Arguments
        ---------
        - var_name -- Str; Precip accum variable name.
        """
        field_dict = self.var_lookup['precipitation']
        section = field_dict['stash_section']
        item = field_dict['stash_item']
        cf1 = lambda cube: \
            cube.attributes['STASH'].section == section and \
            cube.attributes['STASH'].item == item
        coord_constraint_dict = {}
        if time_ix != ForestDataset.TIME_INDEX_ALL:
            period_start_ix = time_ix - (window_size/2.0) - TIME_EPSILON_HRS
            period_start = datetime.datetime.utcfromtimestamp(period_start_ix * 3600)
            period_end_ix = time_ix + (window_size/2.0) + TIME_EPSILON_HRS
            period_end = datetime.datetime.utcfromtimestamp(period_end_ix * 3600)

            def time_window_extract(start1, end1, cell1):
                return cell1.bound[0] >= start1 and cell1.bound[1] <= end1
            coord_constraint_dict['time'] = functools.partial(time_window_extract,
                                                              period_start,
                                                              period_end)

        ic1 = iris.Constraint(cube_func=cf1,
                              coord_values=coord_constraint_dict)

        path_to_load = self.file_loader.load_file(self.file_name)
        precip_cube1 = iris.load_cube(path_to_load, ic1)
        precip_cube1.convert_units(UNIT_DICT['precipitation']) # convert to average hourly accumulation
        precip_cube1.data *= 3 #multiply by 3 to get 3 hour accumulation
        if precip_cube1.coord('time').shape[0] == 1:
            accum_cube = precip_cube1
        else:
            accum_cube = precip_cube1.collapsed('time', iris.analysis.SUM)
        accum_cube.units = cf_units.Unit(PRECIP_UNIT_ACCUM)
        return accum_cube

    def _basic_ts_load(self, var_name, selected_point):
        field_dict = self.var_lookup[var_name]
        time_ix = ForestDataset.TIME_INDEX_ALL
        dc1 = None
        if self.times[var_name] is None:
            self.load_times(var_name)

        if self.data[var_name][time_ix] is None:
            path_to_load = self.file_loader.load_file(self.file_name)
            stash_item = field_dict['stash_item']
            stash_section = field_dict['stash_section']
            dc1 = do_cube_load(path_to_load,
                               stash_section,
                               stash_item,
                               time_ix=time_ix,
                               lat_long_coord=selected_point)
        return dc1


    def _wind_ts_loader(self, var_name, selected_point):
        cube_x_wind = self._basic_ts_load('x_wind',
                                   selected_point)
        cube_y_wind = self._basic_ts_load('x_wind',
                                   selected_point)
        if cube_x_wind is None or cube_y_wind is None:
            return None
        cube_pow = iris.analysis.maths.exponentiate
        ws_cube = cube_pow(cube_pow(cube_x_wind, 2.0) +
                                              cube_pow(cube_y_wind, 2.0),
                                              0.5)
        ws_cube.rename(WIND_SPEED_NAME)
        return ws_cube


    def _precip_accum_ts_load(self, ws1, var_name, selected_point):
        accum_precip_cube = None
        precip_cube = self._basic_ts_load(PRECIP_VAR_NAME,
                                          selected_point,
                                          )
        # cube may be none if this config has no data for the selected lat/lon
        if precip_cube is None:
            return None

        # convert to average hourly accumulation in mm
        precip_cube.convert_units(UNIT_DICT['precipitation'])
        # multiply by 3 to get 3 hour accumulation
        precip_cube.data *= 3

        # create a new coordinate for accumlating precip
        def conv_func_raw(window_length, coord, value):
            return value - value % window_length
        conv_func = functools.partial(conv_func_raw,ws1)
        iris.coord_categorisation.add_categorised_coord(
            cube=precip_cube,
            name='agg_time',
            from_coord='time',
            category_function=conv_func,
            units=precip_cube.units)

        # aggregate by the new coordinate using sum
        accum_precip_cube = \
            precip_cube.aggregated_by(['agg_time'], iris.analysis.SUM)
        accum_precip_cube.units = cf_units.Unit(PRECIP_UNIT_ACCUM)
        return accum_precip_cube

    def get_timeseries(self, var_name, selected_point, convert_units=True):
        """Calls functions to retrieve and load data.

        Arguments
        ---------
        - var_name -- Str; Redundant: used to match other loaders.

        """
        print('load time series for point ({0},{1})'.format(selected_point[0],
                                                            selected_point[1]))
        # extract the relevant timeseries
        dc1 = self.ts_loaders[var_name](var_name,
                                        selected_point)
        if not dc1:
            return None
        if not check_bounds(dc1,
                            selected_point):
            return None
        interp_pt = [('latitude',selected_point[0]),
                     ('longitude',selected_point[1])]
        scheme1 = iris.analysis.Linear()
        time_series_cube = dc1.interpolate(interp_pt,
                                           scheme1)
        if convert_units:
            if UNIT_DICT[var_name]:
                time_series_cube.convert_units(UNIT_DICT[var_name])
        return time_series_cube
