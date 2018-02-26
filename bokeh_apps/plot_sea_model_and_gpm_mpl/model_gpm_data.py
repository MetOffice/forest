import os
import math
import numpy
import datetime as dt

import iris
import cf_units
import iris.coord_categorisation

import forest.util
import forest.data


def conv_func(coord, value):
    
    '''Create a new time coordinate in an iris cube, with values
     to the previous hour divisible by three, i.e. 04:30 becomes 03:00.

    '''

    return math.floor(value / 3.) * 3


def half_hour_rate_to_accum(data, axis=0):
    
    '''Convert half hour rain rates into accumulations

    '''

    accum_array = numpy.sum(data, axis=0) / 2

    return accum_array


class GpmDataset(object):

    def __init__(self,
                 config,
                 file_name_list,
                 s3_base,
                 s3_local_base,
                 use_s3_mount,
                 base_local_path,
                 do_download,
                 times_list,
                 fcast_hour,
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
        self.s3_url_list = [os.path.join(self.s3_base, fn1) for fn1 in self.file_name_list]
        self.local_path_list = [os.path.join(self.base_local_path, fn1) for fn1 in self.file_name_list]
        self.raw_data = None
        self.times_list = times_list
        self.fcast_hour = fcast_hour
        self.data = dict([(v1,None) for v1 in forest.data.VAR_NAMES])

        self.retrieve_data()
        self.load_data()


    def __str__(self):
        
        '''
        
        '''
        
        return 'GPM  dataset'

    def get_data(self, var_name):
        
        '''
        
        '''
        
        return self.data[var_name]

    def retrieve_data(self):
        
        '''
        
        '''
        
        if self.do_download:
            if not (os.path.isdir(self.base_local_path)):
                print('creating directory {0}'.format(self.base_local_path))
                os.makedirs(self.base_local_path)
            for s3_url, local_path in zip(self.s3_url_list, self.local_path_list):
                forest.util.download_from_s3(s3_url, local_path)

    def load_data(self):
        
        """
        
        """
        
        self.raw_data = {}

        for file_name, cube_tim_str in zip(self.local_path_list,
                                           self.times_list):

            if os.path.isfile(file_name):
                print('loading data from file {0}'.format(file_name))
                cube_list = iris.load(file_name)
            else:
                continue
            self.raw_data.update({cube_tim_str: cube_list[0]})

        ACCUM = iris.analysis.Aggregator('half_hour_rate_to_accum',
                                         half_hour_rate_to_accum,
                                         units_func=lambda units: 1)

        temp_cube_list = iris.cube.CubeList()
        
        for time1 in self.raw_data.keys():
            
            print('aggregating time {0}'.format(time1))

            raw_cube = self.raw_data[time1]
            iris.coord_categorisation.add_categorised_coord(raw_cube, 'agg_time', 'time', conv_func,
                                                            units=cf_units.Unit('hours since 1970-01-01',
                                                                                 calendar='gregorian'))
            accum_cube = raw_cube.aggregated_by(['agg_time'], ACCUM)
            temp_cube_list.append(accum_cube)

        self.data['precipitation'] = temp_cube_list.concatenate_cube()
        self.data['accum_precip_3hr'] = self.data['precipitation']
        
        # Create 6, 12 and 24 accumulations
        for accum_step in [6, 12, 24]:
            
            var_name = 'accum_precip_{}hr'.format(accum_step)
            temp_cube = self.data['accum_precip_3hr']
            
            agg_lambda = lambda coord, value: math.floor(value / float(accum_step)) * accum_step
            
            def agg_func(coord, value):
    
                '''Create aggregate time by rounding time down to nearest
                accumulation step

                '''

                return math.floor(value / float(accum_step)) * accum_step
 
            agg_var_name = 'agg_time_' + str(accum_step)

            iris.coord_categorisation.add_categorised_coord(temp_cube, agg_var_name, 'agg_time', agg_lambda,
                                                            units=cf_units.Unit('hours since 1970-01-01',
                                                                                 calendar='gregorian'))
            self.data[var_name] = temp_cube.aggregated_by([agg_var_name], iris.analysis.SUM)

        # Cut out the first 12 hours of data if using a 12Z run
        if self.fcast_hour == 12:
            midday_time_float = self.data['precipitation'].coords('time')[0].points[0]
            print('xxxx', midday_time_float)
            print('xxxx', dt.datetime(1970, 1, 1) + dt.timedelta(hours=int(midday_time_float)))
            midday_constraint = iris.Constraint(time=lambda t: t.point.day >= 12)
            for accum_key in [key for key in self.data.keys() if 'accum' in key]:
                self.data[accum_key] = self.data[accum_key].extract(midday_constraint)