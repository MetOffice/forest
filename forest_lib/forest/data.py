"""
Module containing a class to manage the datasets for the model evaulation tool.
In particular, the data class supports just in time loading.
"""
import os
import datetime
import configparser
import functools

import numpy

import iris

import forest.util

import pdb
WIND_SPEED_NAME = 'wind_speed'
WIND_VECTOR_NAME = 'wind_vectors'
WIND_STREAM_NAME = 'wind_streams'
WIND_MSLP_NAME = 'wind_mslp'
WIND_UNIT_MPH = 'miles-hour^-1'

VAR_NAMES = ['precipitation',
             'air_temperature',
             WIND_SPEED_NAME,
             WIND_VECTOR_NAME,
             WIND_MSLP_NAME,
             'cloud_fraction',
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

UNIT_DICT = {'precipitation': 'kg-m-2-hour^-1',
             'cloud_fraction': None,
             'air_temperature': 'celsius',
             'x_wind': WIND_UNIT_MPH,
             'y_wind': WIND_UNIT_MPH,
             WIND_SPEED_NAME: WIND_UNIT_MPH,
             'mslp': 'hectopascals',
             WIND_VECTOR_NAME: WIND_UNIT_MPH,
             WIND_STREAM_NAME: WIND_UNIT_MPH,
             WIND_MSLP_NAME: WIND_UNIT_MPH,
             }
UNIT_DICT.update(dict([(var1,WIND_UNIT_MPH) for var1 in WIND_VECTOR_VARS]))

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
    available_times = datasets[forest.data.N1280_GA6_KEY]['data'].get_times(var1)

    for ds_name in datasets:
        times1 = datasets[ds_name]['data'].get_times(var1)
        available_times = numpy.array([t1 for t1 in available_times if t1 in times1])
    return available_times

class ForestDataset(object):

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



        # set up data loader functions
        self.loaders = dict([(v1, self._basic_cube_load) for v1 in VAR_NAMES])
        self.loaders[WIND_SPEED_NAME] = self._wind_speed_loader
        self.loaders[WIND_VECTOR_NAME] = self._wind_vector_loader
        for wv_var in WIND_VECTOR_VARS:
            self.loaders[wv_var] = self._wind_vector_loader

        self.data = dict([(v1, None) for v1 in self.loaders.keys()])
        self.times = dict([(v1, None) for v1 in self.loaders.keys()])

        if self.use_s3_local_mount:
            self.path_to_load = self.s3_local_path
        else:
            self.path_to_load = self.local_path

    def __str__(self):
        return 'FOREST dataset'

    def check_data(self):
        """
        Check that the data represented by this dataset actually exists.
        """
        file_exists = False
        if self.do_download:
            file_exists = forest.util.check_remote_file_exists(self.s3_url)
        else:
            file_exists = os.path.isfile(self.path_to_load)
        return file_exists

    def get_times(self, var_name):
        # if self.times[var_name] is None:

        return self.load_times(var_name)

    def load_times(self, var_name):
        return self.time_loaders[var_name](var_name)

    def _basic_time_load(self, var_name):
        field_dict = self.var_lookup[var_name]
        cf1 = lambda cube1: \
            cube1.attributes['STASH'].section == \
            field_dict['stash_section'] and \
            cube1.attributes['STASH'].item == \
            field_dict['stash_item']

        ic1 = iris.Constraint(cube_func=cf1)

        cube1 = iris.load_cube(self.path_to_load, ic1)
        # self.times[var_name] = cube1.coord('time').points
        # self.data[var_name] = dict([(t1,None) for t1 in self.times[var_name]])
        return cube1.coord('time').points

    def _wind_time_load(self):
        return self._basic_time_load('x_wind')
        # self.times['y_wind'] = self.times['y_wind']
        # for var1 in WIND_VECTOR_VARS:
        #     self.times[var1] = self.times['x_wind']

    def get_data(self, var_name, convert_units=True, selected_time=None):
        print('ForestData.get_data 1')
        time_ix = selected_time
        if time_ix is None:
            time_ix = ForestDataset.TIME_INDEX_ALL
        else:
            print('loading data for time {0}'.format(time_ix))

        print('ForestData.get_data 2')

        # first time we look at this data, populate dictionary with
        # available times for this variable
        times1 = None
        if self.data[var_name] is None:
            print('ForestData.get_data 3-1')
            if self.check_data():
                print('ForestData.get_data 3-2')
                # get data from aws s3 storage
                self.retrieve_data()
                print('ForestData.get_data 3-3')
                # load the data into memory from file (will only load meta data
                # initially)
                times1 = self.load_times(var_name)
                print('ForestData.get_data 3-4')

        print('ForestData.get_data 4')
        dc1 = None
        # if self.data[var_name][time_ix] is None:
        #     print('ForestData.get_data 4-1')
        if self.check_data():
            print('ForestData.get_data 4-2')
            # get data from aws s3 storage
            self.retrieve_data()
            print('ForestData.get_data 4-3')
            # load the data into memory from file (will only load meta data
            # initially)
            dc1 = self.load_data(var_name, time_ix)
            print('ForestData.get_data 4-4')

        # else:
        #     self.data[var_name][time_ix] = None

        print('ForestData.get_data 5')

        if dc1 and convert_units and UNIT_DICT[var_name]:
            try:
                print('ForestData.get_data 6-1')
                if dc1.units != UNIT_DICT[var_name]:
                    if UNIT_DICT[var_name]:
                        print('ForestData.get_data 6-2')
                        dc1.convert_units(UNIT_DICT[var_name])
                        print('unit conversion applied to {0}'.format(
                            var_name))
            except (KeyError,AttributeError):
                print('unit conversion not a applicable to {0}'.format(var_name))

        print('ForestData.get_data 7')
        return dc1

    def retrieve_data(self):
        '''
        '''
        if self.do_download:
            if not (os.path.isdir(self.base_local_path)):
                print('creating directory {0}'.format(self.base_local_path))
                os.makedirs(self.base_local_path)

            forest.util.download_from_s3(self.s3_url, self.local_path)

    def load_data(self, var_name,time_ix):
        return self.loaders[var_name](var_name, time_ix)

    def _basic_cube_load(self, var_name, time_ix):
        print('FortestData._basic_cube_load 1')
        time_obj = datetime.datetime.fromtimestamp(time_ix*3600)
        field_dict = self.var_lookup[var_name]
        cf1 = lambda cube1: \
            cube1.attributes['STASH'].section == \
            field_dict['stash_section'] and \
            cube1.attributes['STASH'].item == \
            field_dict['stash_item']
        coord_constraint_dict = {}

        if int(iris.__version__.split('.')[0]) == 1:
            def time_comp(time_index, eps1, cell1):
                return abs(cell1.point - time_index) < eps1

            print('FortestData._basic_cube_load 2')
            if time_ix != ForestDataset.TIME_INDEX_ALL:
                coord_constraint_dict['time'] = \
                    functools.partial(time_comp, time_ix, 1e-5)

        elif int(iris.__version__.split('.')[0]) == 2:
            def time_comp(selected_time, eps1, cell1):
                return abs(cell1.point - selected_time).total_seconds() < eps1

            print('FortestData._basic_cube_load 2')
            if time_ix != ForestDataset.TIME_INDEX_ALL:
                coord_constraint_dict['time'] = \
                    functools.partial(time_comp, time_obj, 1)

        ic1 = iris.Constraint(cube_func=cf1,
                              coord_values=coord_constraint_dict)

        print('FortestData._basic_cube_load 3')

        print('path to load {0}'.format(self.path_to_load))
        print('time to load {0}'.format(str(time_obj)))
        print('stash to load section {0} item {1}'.format(field_dict['stash_section'], field_dict['stash_item'] ) )

        # pdb.set_trace()
        dc1 = iris.load_cube(self.path_to_load, ic1)

        print('FortestData._basic_cube_load 4')
        return dc1

        print('ForestData._basic_cube_load 5')


    def _wind_speed_loader(self, var_name, time_ix):
        # process wind cubes to calculate wind speed

        cube_pow = iris.analysis.maths.exponentiate
        print('calculating wind speed for {0}'.format(self.config_name))
        cube_x_wind = self.get_data('x_wind', time_ix)
        cube_y_wind = self.get_data('y_wind', time_ix)

        self.data[WIND_SPEED_NAME][time_ix] = cube_pow(cube_pow(cube_x_wind, 2.0) +
                                              cube_pow(cube_y_wind, 2.0),
                                              0.5)
        self.data[WIND_SPEED_NAME][time_ix].rename(WIND_SPEED_NAME)

    def _wind_vector_loader(self, var_name, time_ix):
        cube_x_wind = self.get_data('x_wind', time_ix)
        cube_y_wind = self.get_data('y_wind', time_ix)

        wv_dict = forest.util.calc_wind_vectors(cube_x_wind,
                                                cube_y_wind,
                                                10)
        for var1 in wv_dict:
            self.data[var1][time_ix] = wv_dict[var1]
