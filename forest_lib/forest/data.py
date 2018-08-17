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
import configparser
import functools
import numpy
import copy
import dateutil.parser

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

N1280_GA6_KEY = 'n1280_ga6'
KM4P4_RA1T_KEY = 'km4p4_ra1t'
KM1P5_INDO_RA1T_KEY = 'indon2km1p5_ra1t'
KM1P5_MAL_RA1T_KEY = 'mal2km1p5_ra1t'
KM1P5_PHI_RA1T_KEY = 'phi2km1p5_ra1t'
GA6_CONF_ID = 'ga6'
RA1T_CONF_ID = 'ra1t'

VAR_LIST_DIR = os.path.dirname(__file__)
VAR_LIST_FNAME_BASE = 'var_list_{config}.conf'


def get_var_lookup(config):

    """Read config files into dictionary.
    
    Arguments
    ---------
    
    - config -- Str; set config type to read file for.
    
    """
    
    var_list_path = os.path.join(VAR_LIST_DIR,
                                 VAR_LIST_FNAME_BASE.format(config=config))
    parser1 = configparser.RawConfigParser()
    parser1.read(var_list_path)
    field_dict = {}
    for sect1 in parser1.sections():
        field_dict[sect1] = dict(parser1.items(sect1))
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


def get_model_run_times(days_since_period_start, num_days, model_run_period):
    """Create a list of model times from the last num_days days.

    Arguments
    ---------

    - num_days -- Int; Set number of days to go back and get dates for.
    - model_run_period -- Int; period of model runs in hours i.e. their is a model run every model_run_period hours.

    """
    period_start = datetime.datetime.now() + datetime.timedelta(days=-days_since_period_start)
    ps_mn_str = '{dt.year:04}{dt.month:02}{dt.day:02}T0000Z'.format(
        dt=period_start)
    ps_midnight = dateutil.parser.parse(str(ps_mn_str))
    forecast_datetimes = [
        ps_midnight + datetime.timedelta(hours=step1)
        for step1 in range(0, num_days * NUM_HOURS_IN_DAY, model_run_period)]
    return forecast_datetimes


def format_model_run_times(forecast_datetimes):
    fmt_str = '{dt.year:04}{dt.month:02}{dt.day:02}' + \
              'T{dt.hour:02}{dt.minute:02}Z'
    return [fmt_str.format(dt=dt1) for dt1 in forecast_datetimes]


def get_available_datasets(file_loader,
                           dataset_template,
                           days_since_period_start,
                           num_days,
                           model_period,
                           ):
    """
    Get a list of model runs times for which there is model output data
    available, and list of dictionaries with a dataset for each model run time.

    :param file_loader: class responsible for loading remote or local files
    :param dataset_template: A dictionary of dataset configs, which will be
                             used a template for each model run that is
                             available.
    :param days_since_period_start: Number of days in the past to start
                                    looking for model runs
    :param num_days: number of days of model runs to look for
    :param model_period: The period between model runs.
    :return: A tuple containing a list of available model runs and a list of
             datasets for each available model run
    """
    fcast_dt_list = get_model_run_times(days_since_period_start,
                                        num_days,
                                        model_period)
    fcast_dt_str_list = format_model_run_times(fcast_dt_list)
    fcast_time_list = []
    datasets = {}
    for fct, fct_str in zip(fcast_dt_list, fcast_dt_str_list):

        fct_data_dict = copy.deepcopy(dict(dataset_template))
        model_run_data_present = True
        for ds_name in dataset_template.keys():
            file_name = 'model_data/SEA_{conf}_{fct}.nc'.format(conf=ds_name, fct=fct_str)
            fct_data_dict[ds_name]['data'] = forest.data.ForestDataset(file_name,
                                                                       file_loader,
                                                                       dataset_template[ds_name]['var_lookup'])

            model_run_data_present = model_run_data_present and file_loader.file_exists(file_name)
        # include forecast if all configs are present
        # TODO: reconsider data structure to allow for some model configs at different times to be present
        if model_run_data_present:
            datasets[fct_str] = fct_data_dict
            fcast_time_list += [fct_str]

    # select most recent available forecast
    try:
        fcast_time = fcast_time_list[-1]
    except IndexError:
        fcast_time = None
    return fcast_time, datasets


def get_available_times(datasets, var1):
    key0 = list(datasets.keys())[0]
    available_times = datasets[key0]['data'].get_times(var1)
    for ds_name in datasets:
        times1 = datasets[ds_name]['data'].get_times(var1)
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

def do_cube_load(path_to_load,
                 field_dict,
                 time_ix,
                 lat_long_coord):
    """
    Load a cube from the given path using the specified constraints. The field
    dict argument is required, but the time_ix and lat_long_coord arguments are
    optional. The cube loaded will be constrained by one or both of time and
    lat/lon if specified.
    :param path_to_load: The file path to load
    :param field_dict: A dictionary containing keys stash_section and
                      stash_item for selecting a variable from the data in the
                      file.
    :param time_ix: A float representing the time in hours since 1970. Only
                    this time coordinate will be loaded if specified, otherwise
                    all times for the specified variable will be loaded.
    :param lat_long_coord: a lat long coordinate around which to load a half
                           degree window of points.
    :return: The iris cube of the data at the path with given constraints.
    """
    cf1 = lambda cube1: \
        cube1.attributes['STASH'].section == \
        field_dict['stash_section'] and \
        cube1.attributes['STASH'].item == \
        field_dict['stash_item']
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
            def time_comp(selected_time, eps1, cell1):

                return abs(cell1.point - selected_time).total_seconds() < eps1


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
    print('stash to load section {0} item {1}'.format(field_dict['stash_section'],
                                                      field_dict['stash_item']))

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

    def __init__(self,
                 file_name,
                 file_loader,
                 var_lookup):
        """ForestDataset factory function"""
        self.var_lookup = var_lookup
        self.file_name = file_name
        self.file_loader = file_loader

        # set up time loaders
        self.time_loaders = dict([(v1, self._basic_time_load) for v1 in VAR_NAMES])
        for wv_var in WIND_VARS:
            self.time_loaders[wv_var] = self._wind_time_load
        for accum_precip_var in PRECIP_ACCUM_VARS:
            ws1 = PRECIP_ACCUM_WINDOW_SIZES_DICT[accum_precip_var]
            self.time_loaders[accum_precip_var] = functools.partial(self._accum_precip_time_load, ws1)
        self.times = dict([(v1, None) for v1 in self.time_loaders.keys()])


        # set up data loader functions
        self.cube_loaders = dict([(v1, self.basic_cube_load) for v1 in VAR_NAMES])
        self.cube_loaders[WIND_SPEED_NAME] = self.wind_speed_loader
        self.cube_loaders[WIND_STREAM_NAME] = self.wind_speed_loader

        for wv_var in WIND_VECTOR_VARS:
            self.cube_loaders[wv_var] = self.wind_vector_loader
        for accum_precip_var in PRECIP_ACCUM_VARS:
            ws1 = PRECIP_ACCUM_WINDOW_SIZES_DICT[accum_precip_var]
            self.cube_loaders[accum_precip_var] = functools.partial(self.accum_precip_loader, ws1)

        self.data = dict([(v1, None) for v1 in self.cube_loaders.keys()])

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

    def get_times(self, var_name):
        self.load_times(var_name)
        return self.times[var_name]

    def load_times(self, var_name):
        self.time_loaders[var_name](var_name)

    def _basic_time_load(self, var_name):
        """
        Has several responsibilities:

        * Constrains cube to be only variables with correct
          stash section and item codes
        * Populates self.times[var_name] = [t0, t1, ...]
        * Populates self.data[var_name] = {t0: None, t1: None, ...}
        """
        path_to_load = self.file_loader.load_file(self.file_name)
        field_dict = self.var_lookup[var_name]
        cf1 = lambda cube1: \
            cube1.attributes['STASH'].section == \
            field_dict['stash_section'] and \
            cube1.attributes['STASH'].item == \
            field_dict['stash_item']
        ic1 = iris.Constraint(cube_func=cf1)
        cube1 = iris.load_cube(path_to_load, ic1)
        self.times[var_name] = cube1.coord('time').points
        self.data[var_name] =  dict([(t1,None) for t1 in self.times[var_name]] + [('all',None)])

    def _wind_time_load(self, var_name):
        """
        """
        if self.times['x_wind'] is None:
            self._basic_time_load('x_wind')
            self.data['x_wind'].update(dict([(t1,None,) for t1 in self.times['x_wind'] ]))

        for var1 in WIND_VARS:
            if self.times[var1] is None:
                self.times[var1] = copy.deepcopy(self.times['x_wind'])
                self.data[var1] = dict([(t1,None,) for t1 in self.times[var1] ]+ [('all',None)])

    def _accum_precip_time_load(self, window_size1, var_name):
        """
        """
        self._basic_time_load(PRECIP_VAR_NAME)
        self.times[var_name] = \
            numpy.unique(numpy.floor(self.times[PRECIP_VAR_NAME] / window_size1) * window_size1) + (window_size1/2.0)
        self.data[var_name] = dict([(t1, None) for t1 in self.times[var_name]] + [('all',None)])

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

        if self.times[var_name] is None:
            self.load_times(var_name)

        if self.data[var_name][selected_time] is None:
            if self.file_loader.file_exists(self.file_name):
                # Load the data into memory from file (will only load
                # metadata initially)
                self.load_data(var_name, time_ix)
                has_units = \
                    self.data[var_name][time_ix].units is not None and \
                    self.data[var_name][time_ix].units.name != 'unknown'
                if convert_units and has_units:
                    if UNIT_DICT[var_name]:
                        self.data[var_name][time_ix].convert_units(
                            UNIT_DICT[var_name])
            else:
                self.data[var_name] = None
        return self.data[var_name][time_ix]

    def load_data(self, var_name, selected_time):
        """Call loader function.

        Arguments
        ---------
        - var_name -- Str; Var name used as dict key to select loader.
        """
        self.cube_loaders[var_name](var_name, selected_time)

    def basic_cube_load(self, var_name, time_ix):
        """Load simple cubes.

        Arguments
        ---------
        
        - var_name -- Str; Var name used to define data to load.
        - time_ix -- Index of the time to load

        """
        field_dict = self.var_lookup[var_name]
        path_to_load = self.file_loader.load_file(self.file_name)
        dc1 = do_cube_load(path_to_load=path_to_load,
                           field_dict=field_dict,
                           time_ix=time_ix,
                           lat_long_coord=None)
        self.data[var_name][time_ix] = dc1

    def wind_speed_loader(self, var_name, time_ix):
    
        """Process wind cubes to calculate wind speed.
        
        Arguments
        ---------
        
        - var_name -- Str; Redundant: used to match other loaders.
        - time_ix -- Str; specify the time to load
        
        """

        cube_pow = iris.analysis.maths.exponentiate
        print('calculating wind speed')
        cube_x_wind = self.get_data('x_wind', time_ix)
        cube_y_wind = self.get_data('y_wind', time_ix)

        self.data[WIND_SPEED_NAME][time_ix] = cube_pow(cube_pow(cube_x_wind, 2.0) +
                                              cube_pow(cube_y_wind, 2.0),
                                              0.5)
        self.data[WIND_SPEED_NAME][time_ix].rename(WIND_SPEED_NAME)

    def wind_vector_loader(self, var_name, time_ix):
    
        """Gets wind data and calculates wind vectors.

        Arguments
        ---------
        
        - var_name -- Str; Redundant: used to match other loaders.

        """
        
        cube_x_wind = self.get_data('x_wind', time_ix)
        cube_y_wind = self.get_data('y_wind', time_ix)
        wv_dict = forest.util.calc_wind_vectors(cube_x_wind,
                                                cube_y_wind,
                                                WIND_GRID_SIZE)
        for var1 in wv_dict:
            self.data[var1][time_ix] = wv_dict[var1]
        
    def accum_precip_loader(self, window_size, var_name, time_ix):
        
        """Gets data and creates accumulated precipitation cube.

        Arguments
        ---------
        
        - var_name -- Str; Precip accum variable name.

        """
        field_dict = self.var_lookup['precipitation']
        cf1 = lambda cube1: \
            cube1.attributes['STASH'].section == \
            field_dict['stash_section'] and \
            cube1.attributes['STASH'].item == \
            field_dict['stash_item']
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

        self.data[var_name][time_ix] = accum_cube

    def _basic_ts_load(self, var_name, selected_point):
        field_dict = self.var_lookup[var_name]
        time_ix = ForestDataset.TIME_INDEX_ALL
        dc1 = None
        if self.times[var_name] is None:
            self.load_times(var_name)

        if self.data[var_name][time_ix] is None:
            path_to_load = self.file_loader.load_file(self.file_name)
            dc1 = do_cube_load(path_to_load=path_to_load,
                               field_dict=field_dict,
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
                                        selected_point,
                                        )

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
