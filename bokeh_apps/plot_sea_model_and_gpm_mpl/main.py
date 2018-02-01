### SE Asia Model and GPM IMERG Matplotlib example Bokeh app script

# This script demonstrates creating plots of model rainfall data and GPM IMERG
#  data for SE Asia using the Matplotlib plotting library to provide images to
#  a Bokeh Server App.

## Setup notebook
# Do module imports

import os
import datetime

import numpy

import matplotlib
matplotlib.use('agg')

import iris
iris.FUTURE.netcdf_promote = True

import bokeh.plotting


import sea_plot
import sea_control

import lib_sea

## Extract
# Extract data from S3. The data can either be downloaded in full before 
#  loading, or downloaded on demand using the /s3 filemount. This is 
#  controlled by the do_download flag.

bokeh_id = __name__

bucket_name = 'stephen-sea-public-london'
server_address = 'https://s3.eu-west-2.amazonaws.com'

fcast_time = '20180109T0000Z'
fcast_time_obj =  datetime.datetime.strptime(fcast_time, '%Y%m%dT%H%MZ')

N1280_GA6_KEY = 'n1280_ga6'
KM4P4_RA1T_KEY = 'km4p4_ra1t'
KM1P5_INDO_RA1T_KEY = 'indon2km1p5_ra1t'
KM1P5_MAL_RA1T_KEY = 'mal2km1p5_ra1t'
KM1P5_PHI_RA1T_KEY = 'phi2km1p5_ra1t'
GPM_IMERG_EARLY_KEY = 'gpm_imerg_early'
GPM_IMERG_LATE_KEY = 'gpm_imerg_late'

# The datasets dictionary is the main container for the forecast data and 
#  associated meta data. It is stored as a dictionary of dictionaries.
# The first layer of indexing selects the region/config, for example N1280 GA6 
#  global or 1.5km Indonesia RA1-T. Each of these will be populated with a 
#  cube for each of the available variables as well as asociated metadata such
#  as model name, file paths etc.

datasets = {N1280_GA6_KEY:{'data_type_name':'N1280 GA6 LAM Model'},
            KM4P4_RA1T_KEY:{'data_type_name':'SE Asia 4.4KM RA1-T '},
            KM1P5_INDO_RA1T_KEY:{'data_type_name':'Indonesia 1.5KM RA1-T'},
            KM1P5_MAL_RA1T_KEY:{'data_type_name':'Malaysia 1.5KM RA1-T'},
            KM1P5_PHI_RA1T_KEY:{'data_type_name':'Philipines 1.5KM RA1-T'},
            GPM_IMERG_EARLY_KEY:{'data_type_name':'GPM IMERG Early'},
            GPM_IMERG_LATE_KEY:{'data_type_name':'GPM IMERG Late'},
            }

# Model data dict population

s3_base_str = '{server}/{bucket}/model_data/'
s3_base = s3_base_str.format(server=server_address, bucket=bucket_name)
s3_local_base = os.path.join(os.sep,'s3',bucket_name, 'model_data')

for ds_name in datasets.keys():
    
    if 'gpm_imerg' in ds_name:
        continue
        
    fname1 = 'SEA_{conf}_{fct}.nc'.format(conf=ds_name,
                                          fct=fcast_time)
    datasets[ds_name]['fnames_list'] = [fname1]
    datasets[ds_name]['s3_url_list'] = [os.path.join(s3_base, fname1)]
    datasets[ds_name]['s3_local_path'] = [os.path.join(s3_local_base, fname1)]
    datasets[ds_name]['data'] = None

# GPM IMERG dict population

s3_base_str = '{server}/{bucket}/gpm_imerg/'
s3_base = s3_base_str.format(server=server_address, bucket=bucket_name)
s3_local_base = os.path.join(os.sep,'s3',bucket_name, 'gpm_imerg')

for ds_name in datasets.keys():
    
    if 'gpm_imerg' not in ds_name:
        continue
    
    imerg_type = ds_name.split('_')[-1]
    imerg_type_key = 'gpm_imerg_' + imerg_type
        
    fname_fmt = 'gpm_imerg_NRT{im}_V05B_{datetime}_sea_only.nc'
    fnames_list = [fname_fmt.format(im = imerg_type, datetime = (fcast_time_obj + \
                                    datetime.timedelta(days = dd)).strftime('%Y%m%d'))
                   for dd in range(4)]
    datasets[imerg_type_key]['fnames_list'] = fnames_list
    datasets[imerg_type_key]['s3_url_list'] = [os.path.join(s3_base, fname) 
                                               for fname in fnames_list]
    datasets[imerg_type_key]['s3_local_path'] = [os.path.join(s3_local_base, fname) 
                                                 for fname in fnames_list]
    datasets[imerg_type_key]['data'] = None

fname_key = 's3_local_path'

# Download the data from S3

# This code downloads the files to the local disk before loading, rather than 
#  using the s3 FS mount. This is for using when
# S3 mount is unavailable, or for performance reasons.
do_download = True
use_jh_paths = True
base_dir = os.path.expanduser('~/SEA_data')
if do_download:
    for dtype in datasets.keys():
        if 'gpm_imerg' not in dtype:
            if use_jh_paths:
                base_path_local = os.path.join(base_dir,'model_data')
            else:
                base_path_local = '/usr/local/share/notebooks/sea_model_data/'
            if not (os.path.isdir(base_path_local)):
                print('creating directory {0}'.format(base_path_local))
                os.makedirs(base_path_local)
        else:
            if use_jh_paths:
                base_path_local = os.path.join(base_dir, 'gpm_imerg') + '/'
            else:
                base_path_local = '/usr/local/share/notebooks/gpm_imerg/'
            if not (os.path.isdir(base_path_local)):
                print('creating directory {0}'.format(base_path_local))
                os.makedirs(base_path_local)
                
        datasets[dtype]['local_paths_list'] = [os.path.join(base_path_local, file_name) 
                                               for file_name in datasets[dtype]['fnames_list']]

        for s3_url, local_path in zip(datasets[dtype]['s3_url_list'], datasets[dtype]['local_paths_list']):
            try:
                lib_sea.download_from_s3(s3_url, local_path)
            except:
                print("    Warning: file not downloaded successfully:", s3_url)

        fname_key = 'local_paths_list'
        

## Load
# Load the data using iris. 
# The following cell sets up various dictionaries to make it easier to refer to
#  the data. In particular, the precipitation variable name varies between 
#  datasets so these dictionaries reduce code duplication by allowing all 
#  datasets to used in the same way.

plot_names = ['precipitation']

var_names = ['precipitation']

datasets[N1280_GA6_KEY]['var_lookup'] = {'precipitation': 'precipitation_flux'}
datasets[N1280_GA6_KEY]['units'] = {'precipitation': 'kg-m-2-hour^-1'}
datasets[KM4P4_RA1T_KEY]['var_lookup'] = {'precipitation': 'stratiform_rainfall_rate'}
datasets[KM4P4_RA1T_KEY]['units'] = {'precipitation': 'kg-m-2-hour^-1'}

datasets[KM1P5_INDO_RA1T_KEY]['units'] = dict(datasets[KM4P4_RA1T_KEY]['units'])
datasets[KM1P5_MAL_RA1T_KEY]['units'] = dict(datasets[KM4P4_RA1T_KEY]['units'])
datasets[KM1P5_PHI_RA1T_KEY]['units'] = dict(datasets[KM4P4_RA1T_KEY]['units'])

datasets[KM1P5_INDO_RA1T_KEY]['var_lookup'] = dict(datasets[KM4P4_RA1T_KEY]['var_lookup'])
datasets[KM1P5_MAL_RA1T_KEY]['var_lookup'] = dict(datasets[KM4P4_RA1T_KEY]['var_lookup'])
datasets[KM1P5_PHI_RA1T_KEY]['var_lookup'] = dict(datasets[KM4P4_RA1T_KEY]['var_lookup'])

# Do actual loading of datasets. Each cube is a single variable from a single 
#  region/config from a single model run. The datasets are saved in a series 
#  of dictionaries described earlier. The key for each variable is the generic 
#  variable name. This may notbe the same as the variable name internal to the 
#  cube.

for ds_name in datasets:
    
    if 'gpm_imerg' in ds_name:
        continue
    
    print('loading dataset {0}'.format(ds_name))
    for var1 in datasets[ds_name]['var_lookup']:
        print('    loading var {0}'.format(var1))
               
        datasets[ds_name][var1] = iris.load_cube(datasets[ds_name][fname_key], 
                                                 datasets[ds_name]['var_lookup'][var1])
        if datasets[ds_name]['units'][var1]:
            datasets[ds_name][var1].convert_units(datasets[ds_name]['units'][var1])
            
for ds_name in datasets:
    
    if 'gpm_imerg' not in ds_name:
        continue
        
    print('loading dataset {0}'.format(ds_name))
    print('    loading var {0}'.format('precipitation'))

    datasets[ds_name]['data'] = {}

    for file_name in datasets[ds_name]['local_paths_list']:
        cube_tim_str = file_name[-20:-12]
        if os.path.isfile(file_name):
            cube_list = iris.load(file_name)
        else: 
            continue
        if 'early' in file_name:
            datasets[ds_name]['data'].update({cube_tim_str: cube_list[0]})
        elif 'late' in file_name:
            datasets[ds_name]['data'].update({cube_tim_str: cube_list[0]})
            

## Aggregate rainfall data to give 3hr accumulation
# Data in the GPM IMERG files is initially represented in 30 minute timesteps.
# We want to be able to plot custom accumulations, which requires summing half-
#  hourly rainfall totals across a custom time range - in this case, 3 hours.

import iris.coord_categorisation
import iris.unit
import math

def conv_func(coord, value):

    ''' A function to create a new time coordinate in an iris cube, with values
     to the previous hour divisible by three, i.e. 04:30 becomes 03:00.
     
    '''

    return math.floor(value/3.)*3
    
def half_hour_rate_to_accum(data, axis = 0):

    ''' A function to convert half hour rain rates into accumulations
    
    '''

    accum_array = numpy.sum(data, axis = 0)/2
    
    return accum_array

ACCUM = iris.analysis.Aggregator('half_hour_rate_to_accum', 
                                 half_hour_rate_to_accum, 
                                 units_func=lambda units: 1)

for ds_name in datasets:
    
    if 'gpm_imerg' not in ds_name:
        continue

    temp_cube_list = iris.cube.CubeList()

    for time in datasets[ds_name]['data'].keys():
        raw_cube = datasets[ds_name]['data'][time]
        iris.coord_categorisation.add_categorised_coord(raw_cube, 'agg_time', 'time', conv_func, 
                                                        units=iris.unit.Unit('hours since 1970-01-01', 
                                                                             calendar='gregorian'))
        accum_cube = raw_cube.aggregated_by(['agg_time'], ACCUM)
        temp_cube_list.append(accum_cube)

    datasets[ds_name]['precipitation'] = temp_cube_list.concatenate_cube()
    
    
## Setup plots
# Set up plot colours and geoviews datasets before creating and showing plots

# create regions dict, for selecting which map region to display
region_dict = {'indonesia': [-15.1, 1.0865, 99.875, 120.111],
               'malaysia': [-2.75, 10.7365, 95.25, 108.737],
               'phillipines': [3.1375, 21.349, 115.8, 131.987],
               'se_asia': [-18.0, 29.96, 90.0, 153.96],
               }

plot_opts = lib_sea.create_colour_opts(plot_names)

# Set the initial values to be plotted
init_time = 12
init_var = 'precipitation'
init_region = 'se_asia'
init_model_left = KM4P4_RA1T_KEY
init_model_right = GPM_IMERG_EARLY_KEY


## Display plots

# Create a plot object for the left model display
plot_obj_left = sea_plot.SEA_plot(datasets,
                                  plot_opts,
                                  'plot_sea_left' + bokeh_id,
                                  init_var,
                                  init_model_left,
                                  init_region,
                                  region_dict,
                                  )

plot_obj_left.current_time = init_time
bokeh_img_left = plot_obj_left.create_plot()

# Create a plot object for the right model display
plot_obj_right = sea_plot.SEA_plot(datasets,
                                   plot_opts,
                                   'plot_sea_right' + bokeh_id,
                                   init_var,
                                   init_model_right,
                                   init_region,
                                   region_dict,
                                   )

plot_obj_right.current_time = init_time
bokeh_img_right = plot_obj_right.create_plot()

plot_obj_right.link_axes_to_other_plot(plot_obj_left)


num_times = 3 * datasets[GPM_IMERG_LATE_KEY]['precipitation'].shape[0]

control1 = sea_control.SEA_control(datasets,
                                   init_time,
                                   num_times,
                                   [plot_obj_left, plot_obj_right],
                                   [bokeh_img_left, bokeh_img_right],)
try:
    bokeh_mode = os.environ['BOKEH_MODE']
except:
    bokeh_mode = 'server'    
    
if bokeh_mode == 'server':
    bokeh.plotting.curdoc().add_root(control1.main_layout)
    bokeh.plotting.curdoc().title = 'Model rainfall vs GPM app'

elif bokeh_mode == 'cli':
    bokeh.io.show(control1.main_layout)
    
