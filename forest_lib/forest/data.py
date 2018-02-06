"""
Module containing a class to manage the datasets for the model evaulation tool.
In particular, the data class supports just in time loading.
"""
import os
import configparser

import iris

import forest.util

WIND_SPEED_NAME = 'wind_speed'
WIND_VECTOR_NAME = 'wind_vectors'
WIND_STREAM_NAME = 'wind_streams'

VAR_NAMES = ['precipitation',
             'air_temperature',
             WIND_SPEED_NAME,
             WIND_VECTOR_NAME,
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
                    'wv_X_grid',
                    ]

UNIT_DICT = {'precipitation': 'kg-m-2-hour^-1',
             'cloud_fraction': None,
             'air_temperature': 'celsius',
             'x_wind': 'miles-hour^-1',
             'y_wind': 'miles-hour^-1',
             WIND_SPEED_NAME: 'miles-hour^-1',
             'mslp': 'hectopascals',
             WIND_VECTOR_NAME: 'miles-hour^-1',
             WIND_STREAM_NAME: 'miles-hour^-1',
             }

N1280_GA6_KEY = 'n1280_ga6'
KM4P4_RA1T_KEY = 'km4p4_ra1t'
KM1P5_INDO_RA1T_KEY = 'indon2km1p5_ra1t'
KM1P5_MAL_RA1T_KEY = 'mal2km1p5_ra1t'
KM1P5_PHI_RA1T_KEY = 'phi2km1p5_ra1t'
GA6_CONF_ID = 'ga6'
RA1T_CONF_ID = 'ra1t'
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


class ForestDataset(object):

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

        # set up data loader functions
        self.loaders = dict([(v1, self.basic_cube_load) for v1 in VAR_NAMES])
        self.loaders[WIND_SPEED_NAME] = self.wind_speed_loader
        self.loaders[WIND_VECTOR_NAME] = self.wind_vector_loader
        for wv_var in WIND_VECTOR_VARS:
            self.loaders[wv_var] = self.wind_vector_loader

        self.data = dict([(v1, None) for v1 in self.loaders.keys()])
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
        return forest.util.check_remote_file_exists(self.s3_url)

    def get_data(self, var_name):
        if self.data[var_name] is None:
            if self.check_data():
                # get data from aws s3 storage
                self.retrieve_data()
                # load the data into memory from file (will only load meta data
                # initially)
                self.load_data(var_name)
            else:
                self.data[var_name] = None

        return self.data[var_name]

    def retrieve_data(self):
        '''
        '''
        if self.do_download:
            if not (os.path.isdir(self.base_local_path)):
                print('creating directory {0}'.format(self.base_local_path))
                os.makedirs(self.base_local_path)

            forest.util.download_from_s3(self.s3_url, self.local_path)

    def load_data(self, var_name):
        self.loaders[var_name](var_name)

    def basic_cube_load(self, var_name):
        field_dict = self.var_lookup[var_name]
        if field_dict['accumulate']:
            cf1 = lambda cube1: \
                cube1.attributes['STASH'].section == \
                field_dict['stash_section'] and \
                cube1.attributes['STASH'].item == \
                field_dict['stash_item'] and \
                len(cube1.cell_methods) > 0
        else:
            cf1 = lambda cube1: \
                cube1.attributes['STASH'].section == \
                field_dict['stash_section'] and \
                cube1.attributes['STASH'].item == \
                field_dict['stash_item'] and \
                len(cube1.cell_methods) == 0

        ic1 = iris.Constraint(cube_func=cf1)

        self.data[var_name] = iris.load_cube(self.path_to_load, ic1)
        if UNIT_DICT[var_name]:
            self.data[var_name].convert_units(UNIT_DICT[var_name])

    def wind_speed_loader(self, var_name):
        # process wind cubes to calculate wind speed

        cube_pow = iris.analysis.maths.exponentiate
        print('calculating wind speed for {0}'.format(self.config_name))
        cube_x_wind = self.get_data('x_wind')
        cube_y_wind = self.get_data('y_wind')

        self.data[WIND_SPEED_NAME] = cube_pow(cube_pow(cube_x_wind, 2.0) +
                                              cube_pow(cube_y_wind, 2.0),
                                              0.5)
        self.data[WIND_SPEED_NAME].rename(WIND_SPEED_NAME)

    def wind_vector_loader(self, var_name):
        cube_x_wind = self.get_data('x_wind')
        cube_y_wind = self.get_data('y_wind')

        self.data.update(forest.util.calc_wind_vectors(cube_x_wind,
                                                       cube_y_wind,
                                                       10))
