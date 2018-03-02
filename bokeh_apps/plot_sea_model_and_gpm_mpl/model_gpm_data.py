"""Module containing a class to manage the datasets for Model vs GPM. 

In particular, the GpmDataset class supports just in time loading.

Functions
---------

- conv_func() -- Rounds times (as floats in hours) down to nearest 3.
- half_hour_rate_to_accum() -- Sums half-hour arrays, divides by 2.

Classes
-------

- ForestDataset -- Main class for containing Forest data.

"""

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
    
    """Round float down to nearest multiple of 3. 
    
    Used to convert times (as floats in hours) to the most recent 
    previous hour divisible by three, i.e. 04:30 becomes 03:00.
    
    Arguments
    ---------
    
    - coord -- Iris coord; Iris coordinate being converted.
    - value -- Float; Value in coordinate to be converted.
    
    """

    return math.floor(value / 3.) * 3


def half_hour_rate_to_accum(data, axis=0):
    
    """Sum array and divide by two.
    
    Used to sum half-hour rain rates and convert to total accumulation.
    
    Arguments
    ---------
    
    - data -- Numpy array; Array to sum over single axis.
    - axis -- Int; Axis to sum over.

    """

    accum_array = numpy.sum(data, axis=0) / 2

    return accum_array


class GpmDataset(object):

    """Declare main class for holding Forest data.
    
    Methods
    -------
    
    - __init__() -- Factory method.
    - __str__() -- String method.
    - get_data() -- Return self.data.
    - retrieve_data() -- Download data from S3 bucket.
    - load_data() -- Read GPM data into Iris cubes, and create accum.
    
    Attributes
    ----------
    
    - config -- Str; Name of data configuration.
    - file_name_list -- List; Specifies netCDF file names.
    - s3_base -- Str; S3 data basepath.
    - s3_local_base -- Str; Local S3 data basepath.
    - use_s3_mount -- Bool; Specify whether to use S3 mount.
    - base_local_path -- Str; Local basepath to data.
    - do_download -- Bool; Specify whether to do data download.
    - s3_url_list -- List; Combined S3 basepath and filenames.
    - local_path_list -- List; Combined local basepath and filenames.
    - raw_data -- Dict; Dict of unaccumulated GPM precip cubes.
    - times_list -- List; List of date strings.
    - data -- Dict; Loaded data cubes.
    
    """
    
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
        
        """GpmDataset factory function"""
        
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
        
        """Return string"""
        
        return 'GPM  dataset'

    def get_data(self, var_name):
        
        """Return data
        
        Arguments
        ---------
        
        - var_name -- Str; Variable name to use as key to self.data
        
        """
        
        return self.data[var_name]

    def retrieve_data(self):
        
        """Download data from S3 bucket."""

        if self.do_download:
            if not (os.path.isdir(self.base_local_path)):
                print('creating directory {0}'.format(self.base_local_path))
                os.makedirs(self.base_local_path)
            for s3_url, local_path in zip(self.s3_url_list, self.local_path_list):
                forest.util.download_from_s3(s3_url, local_path)

    def load_data(self):
        
        """Load data into cubes. Also accumulates precipitation."""
        
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
            iris.coord_categorisation.add_categorised_coord(raw_cube, 
                                                            'agg_time', 
                                                            'time', 
                                                            conv_func,
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
    
                """Create agg time - round time down to nearest accum step"""


                return math.floor(value / float(accum_step)) * accum_step
 
            agg_var_name = 'agg_time_' + str(accum_step)

            iris.coord_categorisation.add_categorised_coord(temp_cube, 
                                                            agg_var_name, 
                                                            'agg_time', 
                                                            agg_lambda,
                                                            units=cf_units.Unit('hours since 1970-01-01',
                                                                                 calendar='gregorian'))
            self.data[var_name] = temp_cube.aggregated_by([agg_var_name], iris.analysis.SUM)

        # Cut out the first 12 hours of data if using a 12Z run
        if self.fcast_hour == 12:
            time_0 = self.data['precipitation'].coords('time')[0].points[0]
            midday_time_int = int(time_0 + 11)
            midday_time_obj = dt.datetime(1970, 1, 1) \
                              + dt.timedelta(hours=midday_time_int)
            midday_constraint = iris.Constraint(time=lambda t: t.point >= midday_time_obj)
            
            for acc_key in [key for key in self.data.keys() if 'accum' in key]:
                self.data[acc_key] = self.data[acc_key].extract(midday_constraint)