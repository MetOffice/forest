"""Module containing a class to manage the datasets for Forest. 

In particular, the ForestDataset class supports just in time loading.

Functions
---------

- get_var_lookup() -- Read config files into dictionary.

Classes
-------

- ForestDataset -- Main class for containing Forest data.

"""

import os
import datetime
import configparser
import functools
import numpy
import math
import copy

import iris
import cf_units

import forest.util

# The number of days into the past to look for data. The current
# value specifies looking for data up to 1 week old
NUM_DATA_DAYS = 7
MODEL_RUN_PERIOD = 12

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
                    ]
PRECIP_ACCUM_WINDOW_SIZES_LIST = [3,6,12,24]

PRECIP_ACCUM_WINDOW_SIZES_DICT = dict([('accum_precip_{0}hr'.format(window1), window1) for window1 in PRECIP_ACCUM_WINDOW_SIZES_LIST])
PRECIP_ACCUM_VARS = list(PRECIP_ACCUM_WINDOW_SIZES_DICT.keys())

UNIT_DICT = {PRECIP_VAR_NAME: PRECIP_UNIT_RATE,
             'cloud_fraction': None,
             'air_temperature': 'celsius',
             'x_wind': WIND_UNIT_MPH,
             'y_wind': WIND_UNIT_MPH,
             WIND_SPEED_NAME: WIND_UNIT_MPH,
             'mslp': 'hectopascals',
             WIND_VECTOR_NAME: WIND_UNIT_MPH,
             WIND_STREAM_NAME: WIND_UNIT_MPH,
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

def get_available_times(datasets, var1):
    key0 = list(datasets.keys())[0]
    available_times = datasets[key0]['data'].get_times(var1)

    for ds_name in datasets:
        times1 = datasets[ds_name]['data'].get_times(var1)
        available_times = numpy.array([t1 for t1 in available_times if t1 in times1])
    return available_times

class ForestDataset(object):

    """Declare main class for holding Forest data.
    
    Methods
    -------
    
    - __init__() -- Factory method.
    - __str__() -- String method.
    - check_data() -- Check data exists.
    - get_data() -- Call self.retrieve_data() and self.load_data().
    - retrieve_data() -- Download data from S3 bucket.
    - load_data() -- Call loader methods.
    - basic_cube_load() -- Load simple data into a cube.
    - wind_speed_loader() -- Load x/y wind data, create speed cube.
    - wind_vector_loader() -- Load x/y wind data, create vector cube.
    - add_accum_precip_keys() -- Add precip. accum. keys to data dict.
    - accum_precip_loader() -- Load precip. data and calc. accums.
    - accum_precip() -- Calculate precip. accumulations.
    
    Attributes
    ----------
    
    - config_name -- Str; Name of data configuration.
    - var_lookup -- Dict; Links variable names to data keys.
    - file_name -- Str; Specifies netCDF file name.
    - s3_base_url -- Str; S3 data basepath.
    - s3_url -- Str; Combined S3 basepath and filename.
    - s3_local_base -- Str; Local S3 data basepath.
    - s3_local_path -- Str; Combined S3 local basepath and filename.
    - use_s3_local_mount -- Bool; Specify whether to use S3 mount.
    - base_local_path -- Str; Local basepath to data.
    - do_download -- Bool; Specify whether to do data download.
    - local_path -- Str; Combined local basepath and filename.
    - loaders -- Dict; Dictionary of loader functions for vars.
    - data -- Dict; Loaded data cubes.
    - path_to_load -- Str; local/S3 path, based on do_download.
    
    """
    TIME_INDEX_ALL = 'all'

    def __init__(self,
                 config,
                 file_name,
                 s3_base,
                 s3_local_base,
                 use_s3_mount,
                 base_local_path,
                 do_download,
                 var_lookup):
        
        """ForestDataset factory function"""
        
        self.config_name = config
        self.var_lookup = var_lookup
        self.file_name = file_name
        self.s3_base_url = s3_base
        self.s3_url = os.path.join(self.s3_base_url, self.file_name)
        self.s3_local_base = s3_local_base
        self.s3_local_path = os.path.join(self.s3_local_base, self.file_name)
        self.use_s3_local_mount = use_s3_mount
        self.base_local_path = base_local_path
        self.do_download = do_download
        self.local_path = os.path.join(self.base_local_path,
                                       self.file_name)

        # set up time loaders


        self.time_loaders = dict([(v1, self._basic_time_load) for v1 in VAR_NAMES])
        self.time_loaders[WIND_SPEED_NAME] = self._wind_time_load
        self.time_loaders[WIND_VECTOR_NAME] = self._wind_time_load
        for wv_var in WIND_VECTOR_VARS:
            self.time_loaders[wv_var] = self._wind_time_load
        for accum_precip_var in PRECIP_ACCUM_VARS:
            self.time_loaders[accum_precip_var] = self._accum_precip_time_load
        self.times = dict([(v1, None) for v1 in self.time_loaders.keys()])


        # set up data loader functions
        self.loaders = dict([(v1, self.basic_cube_load) for v1 in VAR_NAMES])
        self.loaders[WIND_SPEED_NAME] = self.wind_speed_loader
        self.loaders[WIND_VECTOR_NAME] = self.wind_vector_loader
        for wv_var in WIND_VECTOR_VARS:
            self.loaders[wv_var] = self.wind_vector_loader
        for accum_precip_var in PRECIP_ACCUM_VARS:
            self.loaders[accum_precip_var] = self.accum_precip_loader

        self.data = dict([(v1, None) for v1 in self.loaders.keys()])
        if self.use_s3_local_mount:
            self.path_to_load = self.s3_local_path
        else:
            self.path_to_load = self.local_path

    def __str__(self):
    
        """Return string"""
        
        return 'FOREST dataset'

    def check_data(self):
    
        """Check that the data represented by this dataset exists."""

        file_exists = False
        if self.do_download:
            file_exists = forest.util.check_remote_file_exists(self.s3_url)
        else:
            file_exists = os.path.isfile(self.path_to_load)
        return file_exists

    def get_times(self, var_name):
        """
        """
        # if self.times[var_name] is None:
        self.load_times(var_name)
        return self.times[var_name]

    def load_times(self, var_name):
        """
        """
        self.time_loaders[var_name](var_name)

    def _basic_time_load(self, var_name):
        """
        """
        field_dict = self.var_lookup[var_name]
        cf1 = lambda cube1: \
            cube1.attributes['STASH'].section == \
            field_dict['stash_section'] and \
            cube1.attributes['STASH'].item == \
            field_dict['stash_item']

        ic1 = iris.Constraint(cube_func=cf1)

        cube1 = iris.load_cube(self.path_to_load, ic1)
        self.times[var_name] = cube1.coord('time').points
        self.data[var_name] =  dict([(t1,None) for t1 in self.times[var_name]] + [('all',None)])

    def _wind_time_load(self, var_name):
        """
        """
        if self.times['x_wind'] is None:
            self._basic_time_load('x_wind')
            self.data['x_wind'].update(dict([(t1,None,) for t1 in self.times['x_wind'] ]))

            for var1 in WIND_VECTOR_VARS + ['y_wind']:
                self.times[var1] = copy.deepcopy(self.times['x_wind'])
                self.data[var1] = dict([(t1,None,) for t1 in self.times[var1] ]+ [('all',None)])
    def _accum_precip_time_load(self, var_name):
        """
        """
        self._basic_time_load(PRECIP_VAR_NAME)
        window_size1 = PRECIP_ACCUM_WINDOW_SIZES_DICT[var_name]
        self.times[var_name] = numpy.unique(numpy.floor(self.times[PRECIP_VAR_NAME] / window_size1) * window_size1)
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
            if self.check_data():
                # get data from aws s3 storage
                self.retrieve_data()

                self.load_times(var_name)

        if self.data[var_name][selected_time] is None:
            if self.check_data():
                # Get data from aws s3 storage
                self.retrieve_data()
                # Load the data into memory from file (will only load 
                # metadata initially)
                self.load_data(var_name, time_ix)
                if convert_units:
                    if UNIT_DICT[var_name]:
                        self.data[var_name][time_ix].convert_units(UNIT_DICT[var_name])

            else:
                self.data[var_name] = None


        return self.data[var_name][time_ix]

    def retrieve_data(self):
    
        """Download data from S3 bucket."""
        
        if self.do_download:
            if not (os.path.isdir(self.base_local_path)):
                print('creating directory {0}'.format(self.base_local_path))
                os.makedirs(self.base_local_path)

            forest.util.download_from_s3(self.s3_url, self.local_path)

    def load_data(self, var_name, selected_time):
    
        """Call loader function.
        
        Arguments
        ---------
        
        - var_name -- Str; Var name used as dict key to select loader.

        """
        
        self.loaders[var_name](var_name, selected_time)

    def basic_cube_load(self, var_name, time_ix):
    
        """Load simple cubes.

        Arguments
        ---------
        
        - var_name -- Str; Var name used to define data to load.
        - time_ix -- Index of the time to load

        """
        field_dict = self.var_lookup[var_name]
        cf1 = lambda cube1: \
            cube1.attributes['STASH'].section == \
            field_dict['stash_section'] and \
            cube1.attributes['STASH'].item == \
            field_dict['stash_item']
        coord_constraint_dict = {}

        if time_ix != ForestDataset.TIME_INDEX_ALL:
            time_obj = datetime.datetime.fromtimestamp(time_ix * 3600)
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

            ic1 = iris.Constraint(cube_func=cf1,
                                  coord_values=coord_constraint_dict)

        else:
            time_desc = time_ix
            ic1 = iris.Constraint(cube_func=cf1)

        print('path to load {0}'.format(self.path_to_load))
        print('time to load {0}'.format(time_desc))
        print('stash to load section {0} item {1}'.format(field_dict['stash_section'], field_dict['stash_item'] ) )

        dc1 = iris.load_cube(self.path_to_load, ic1)
        self.data[var_name][time_ix] = dc1

    def wind_speed_loader(self, var_name, time_ix):
    
        """Process wind cubes to calculate wind speed.
        
        Arguments
        ---------
        
        - var_name -- Str; Redundant: used to match other loaders.
        - time_ix -- Str; specify the time to load
        
        """

        cube_pow = iris.analysis.maths.exponentiate
        print('calculating wind speed for {0}'.format(self.config_name))
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
                                                10)
        for var1 in wv_dict:
            self.data[var1][time_ix] = wv_dict[var1]
     
    def add_accum_precip_keys(self, timespan):
    
        """Create precipitation accumulation cube dict keys.
        
        Arguments
        ---------
        
        - timespan -- Str; Define precip. accum. window.
        
        """
        
        var_name = 'precip_accum_{}hr'.format(timespan)
        self.data.update({var_name: None})
        
    def accum_precip_loader(self, var_name, time_ix):
        
        """Gets data and creates accumulated precipitation cube.

        Arguments
        ---------
        
        - var_name -- Str; Precip accum variable name.

        """

        self.get_data('precipitation', 'all')
        self.accum_precip(var_name)

    def accum_precip(self, var_name):
    
        """Create precip. accum. cube from existing precip. data.
        
        Arguments
        ---------
        
        - var_name -- Str; Hour in var_name defines accum window.
        
        """
        accum_multiplier = float(var_name[13:-2])
        
        conv_lambda = lambda coord, value: math.floor(value / accum_multiplier) * accum_multiplier 
        
        temp_cube = self.data['precipitation']['all']
        temp_cube.data *= 3
        temp_cube.units = 'kg-m-2'
        
        agg_name = 'agg_time_{}'.format(var_name[13:])
        
        time_unit = cf_units.Unit('hours since 1970-01-01', calendar='gregorian')
        iris.coord_categorisation.add_categorised_coord(temp_cube, 
                                                        agg_name, 
                                                        'time', 
                                                        conv_lambda,
                                                        units=time_unit)
        accum_cube = temp_cube.aggregated_by([agg_name], iris.analysis.SUM)
        accum_cube.coord('time').points = accum_cube.coord(agg_name).points
        self.data[var_name]['all'] = accum_cube
        self.times[var_name] = self.data[var_name]['all'].coord('time').points
        for ix1, time1 in enumerate(self.times[var_name]):
            self.data[var_name][time1] = self.data[var_name]['all'][ix1]