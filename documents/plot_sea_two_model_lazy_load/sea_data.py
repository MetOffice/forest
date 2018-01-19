"""
Module containing a class to manage the datasets for the model evaulation tool. In particular, 
the data class supports just in time loading.
"""
import os 

import numpy

import iris

import lib_sea

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

UNIT_DICT = {'precipitation':'kg-m-2-hour^-1',
             'cloud_fraction': None,
             'air_temperature':'celsius',
             'x_wind':'miles-hour^-1',
             'y_wind':'miles-hour^-1',
             WIND_SPEED_NAME:'miles-hour^-1',
             'mslp':'hectopascals',
             WIND_VECTOR_NAME: 'miles-hour^-1',
             WIND_STREAM_NAME: 'miles-hour^-1',
             }

VAR_LOOKUP_GA6 = {'precipitation':'precipitation_flux',
                                 'cloud_fraction': 'cloud_area_fraction_assuming_maximum_random_overlap',
                                 'air_temperature':'air_temperature',
                                 'x_wind':'x_wind',
                                 'y_wind':'y_wind',
                                 'mslp':'air_pressure_at_sea_level',
                                }

VAR_LOOKUP_RA1T = {'precipitation':'stratiform_rainfall_rate',
                                  'cloud_fraction': 'cloud_area_fraction_assuming_maximum_random_overlap',
                                  'air_temperature':'air_temperature',
                                  'x_wind':'x_wind',
                                  'y_wind':'y_wind',
                                  'mslp':'air_pressure_at_sea_level',
                  }

class SEA_dataset(object):
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
        self.loaders = dict([(v1,self.basic_cube_load) for v1 in VAR_NAMES])
        self.loaders[WIND_SPEED_NAME] = self.wind_speed_loader
        self.loaders[WIND_VECTOR_NAME] = self.wind_vector_loader
        for wv_var in WIND_VECTOR_VARS:
            self.loaders[wv_var] = self.wind_vector_loader
        
        self.data = dict([(v1,None) for v1 in self.loaders.keys()])
        if self.use_s3_local_mount:
            self.path_to_load = self.s3_local_path
        else:
            self.path_to_load = self.local_path

    
    def __str__(self):
        return 'SEA dataset'
    
    def get_data(self, var_name):
        if self.data[var_name] is None:
            self.load_data(var_name)
        
        return self.data[var_name]

    def load_data(self,var_name):
        self.loaders[var_name](var_name)

    def basic_cube_load(self, var_name):
        self.data[var_name] = iris.load_cube(self.path_to_load,
                                             self.var_lookup[var_name])
        if UNIT_DICT[var_name]:
            self.data[var_name].convert_units(UNIT_DICT[var_name])

    def wind_speed_loader(self, var_name):
        # process wind cubes to calculate wind speed
        
        cube_pow = iris.analysis.maths.exponentiate
        print('calculating wind speed for {0}'.format(self.config_name))
        cube_x_wind = self.get_data('x_wind')
        cube_y_wind = self.get_data('y_wind')
        
        self.data[WIND_SPEED_NAME] = cube_pow( cube_pow(cube_x_wind, 2.0) +
                                                    cube_pow(cube_y_wind, 2.0),
                                                    0.5 )
        self.data[WIND_SPEED_NAME].rename(WIND_SPEED_NAME)

    def wind_vector_loader(self, var_name):
        cube_x_wind = self.get_data('x_wind')
        cube_y_wind = self.get_data('y_wind')

        self.data.update(lib_sea.calc_wind_vectors(cube_x_wind,
                                                   cube_y_wind,
                                                   10))

    def retrieve_data(self):
        '''
        '''            
        if self.do_download:
            if not (os.path.isdir(self.base_local_path)):
                print('creating directory {0}'.format(self.base_local_path))
                os.makedirs(self.base_local_path)
            
            for ds_name in datasets:
                lib_sea.download_from_s3(self.s3_url, self.local_path)
                
            
    


