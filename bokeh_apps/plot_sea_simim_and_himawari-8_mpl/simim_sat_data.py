import os
import datetime
import numpy as np
import iris
import matplotlib.pyplot

import forest.data
import forest.util

SIMIM_KEY = 'simim'
HIMAWARI8_KEY = 'himawari-8'

SIMIM_SAT_VARS = ['W', 'V', 'I']

HIMAWARI_KEYS = {'W': 'LWIN11',
                 'V': 'LVIN10',
                 'I': 'LIIN50',
                 }

UNIT_DICT = {'W': None,
             'V': None,
             'I': None,
             }

DATA_TIMESTEPS = {'W' : {0: np.arange(12, 39, 3),
                         12: np.arange(12, 39, 3)},
                  'I' : {0: np.arange(12, 39, 3),
                         12: np.arange(12, 39, 3)},
                  'V' : {0: np.array((24, 27, 30)),
                         12: np.array((18, 36, 39, 42))},
                  }

class SimimDataset(object):
    
    '''
    
    '''
    
    def __init__(self,
                 config,
                 file_name_list,
                 s3_base,
                 s3_local_base,
                 use_s3_mount,
                 base_local_path,
                 do_download,
                 times_list,
                 forecast_time_obj,
                 ):
        self.config = config
        self.file_name_list = file_name_list
        self.times_list = times_list
        self.s3_base = s3_base
        self.s3_local_base = s3_local_base
        self.use_s3_mount = use_s3_mount
        self.base_local_path = base_local_path
        self.do_download = do_download
        self.s3_url_list = [os.path.join(self.s3_base, fn1) for fn1 in self.file_name_list]
        if self.use_s3_mount:
            self.local_path_list = [os.path.join(self.s3_local_base, fn1) for fn1 in self.file_name_list]
        else:
            self.local_path_list = [os.path.join(self.base_local_path, fn1) for fn1 in self.file_name_list]
        self.forecast_time_obj = forecast_time_obj
        self.data = dict([(v1, None) for v1 in SIMIM_SAT_VARS])

        self.retrieve_data()
        self.load_data()

    def __str__(self):
        
        '''
        
        '''
        
        return 'Simulated Imagery dataset'

    def get_times(self, var_name):
        if self.data[var_name] is None:
            self.retrieve_data()
            self.load_data()
        return [k1 for k1 in self.data[var_name].keys()]

    def get_data(self, var_name, convert_units=False, selected_time=None):
        
        '''
        
        '''
        if self.data[var_name] is None:
            self.retrieve_data()
            self.load_data()
        try:
            if selected_time is not None:
                return self.data[var_name][selected_time]
            return self.data[var_name]
        except KeyError as ke1:
            print('data {0} var={1} time={2} not available'.format(self.config, var_name, selected_time))
        return None

    def retrieve_data(self):
        
        '''
        
        '''
        
        if self.do_download:
            if not (os.path.isdir(self.base_local_path)):
                print('creating directory {0}'.format(self.base_local_path))
                os.makedirs(self.base_local_path)
            for s3_url, local_path in zip(self.s3_url_list, self.local_path_list):
                try:
                    forest.util.download_from_s3(s3_url, local_path)
                except:
                    print("    Warning: file not downloaded successfully:", s3_url)
                    
    def load_data(self):
        
        '''
        
        '''
        
        self.data = dict((it1,{}) for it1 in HIMAWARI_KEYS.keys())

        param_number_dict = {'206': 'W', '207': 'X', '208': 'I'}
        
        for file_name in self.local_path_list:
            cube_time_td = datetime.timedelta(hours=int(file_name[-5:-3]))
            cube_time_str = ((self.forecast_time_obj + cube_time_td).strftime('%Y%m%d%H%M'))
            if os.path.isfile(file_name):
                cube_list = iris.load(file_name)
            else:
                continue
            if 'simbt' in file_name:
                param_number_avail = False
                for cube in cube_list:
                    if 'parameterNumber' in cube.attributes.keys():
                        param_number_avail = True
                        param_number = cube.attributes['parameterNumber']
                        if str(param_number) == '207':
                            continue
                        data_type = param_number_dict[str(param_number)]
                        self.data[data_type].update({cube_time_str: cube})
                if not param_number_avail:
                    self.data['W'].update({cube_time_str: cube_list[2]})
                    self.data['I'].update({cube_time_str: cube_list[0]})
            elif 'simvis' in file_name:
                self.data['V'].update({cube_time_str: cube_list[0]})


class SatelliteDataset(object):
    
    '''
    
    '''
    
    def __init__(self,
                 config,
                 file_name_list,
                 s3_base,
                 s3_local_base,
                 use_s3_mount,
                 base_local_path,
                 do_download,
                 ):
        
        '''
        
        '''
        
        self.config = config
        self.file_name_list = file_name_list
        self.s3_base = s3_base
        self.s3_local_base = s3_local_base
        self.use_s3_mount = use_s3_mount
        self.base_local_path = base_local_path
        self.do_download = do_download
        self.s3_url_list = {}
        self.local_path_list = {}
        for im_type in file_name_list.keys():
            self.s3_url_list[im_type] = [os.path.join(self.s3_base, fn1) for fn1 in self.file_name_list[im_type]]
            self.local_path_list[im_type] = [os.path.join(self.base_local_path, fn1) for fn1 in self.file_name_list[im_type]]
            if self.use_s3_mount:
                self.local_path_list[im_type] = [os.path.join(self.s3_local_base, fn1) for fn1 in
                                                 self.file_name_list[im_type]]
            else:
                self.local_path_list[im_type] = [os.path.join(self.base_local_path, fn1) for fn1 in
                                                 self.file_name_list[im_type]]
        self.data = dict([(v1, None) for v1 in SIMIM_SAT_VARS])
        self.retrieve_data()
        self.load_data()

    def __str__(self):
        
        '''
        
        '''
        
        return 'Satellite Image dataset'

    def get_times(self, var_name):
        if self.data[var_name] is None:
            self.retrieve_data()
            self.load_data()

        return [k1 for k1 in self.data[var_name].keys()]

    def get_data(self, var_name, convert_units=False, selected_time=None):
        
        '''
        
        '''
        if self.data[var_name] is None:
            self.retrieve_data()
            self.load_data()

        try:
            if selected_time is not None:
                return self.data[var_name][selected_time]
            return self.data[var_name]
        except KeyError as ke1:
            print('data {0} var={1} time={2} not available'.format(self.config, var_name, selected_time))
        return None

    def retrieve_data(self):
        
        '''
        
        '''
        
        if self.do_download:
            if not (os.path.isdir(self.base_local_path)):
                print('creating directory {0}'.format(self.base_local_path))
                os.makedirs(self.base_local_path)
            for im_type in self.local_path_list.keys():
                for s3_url, local_path in zip(self.s3_url_list[im_type], self.local_path_list[im_type]):
                    print('processing file {0}'.format(s3_url))
                    try:
                        forest.util.download_from_s3(s3_url, local_path)
                    except:
                        print("    Warning: file not downloaded successfully:", s3_url)
                        
    def load_data(self):
        
        '''
        
        '''
        
        self.data = dict((it1,{}) for it1 in self.local_path_list.keys())
        for im_type in self.local_path_list:
            im_array_dict = {}
            for file_name in self.local_path_list[im_type]:
                try:
                    data_time1 = file_name[-16:-4]
                    im_array_dict.update({data_time1: matplotlib.pyplot.imread(file_name)})
                except:
                    print('failed to load file {0}'.format(file_name))
            self.data[im_type].update(im_array_dict)