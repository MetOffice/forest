import os
import time
import sys
import urllib.request
import textwrap
import numpy
import iris

import bokeh.io
import bokeh.layouts 
import bokeh.models.widgets
import bokeh.plotting


import matplotlib
matplotlib.use('agg')

import lib_sea
import plot_tools
from sea_plot import SEA_plot

iris.FUTURE.netcdf_promote = True

try:
    get_ipython
    is_notbook = True
except:
    is_notebook = False


# Extract and Load
bucket_name = 'stephen-sea-public-london'
server_address = 'https://s3.eu-west-2.amazonaws.com'

fcast_time = '20180110T0000Z'

N1280_GA6_KEY = 'n1280_ga6'
KM4P4_RA1T_KEY = 'km4p4_ra1t'
KM1P5_INDO_RA1T_KEY = 'indon2km1p5_ra1t'
KM1P5_MAL_RA1T_KEY = 'mal2km1p5_ra1t'
KM1P5_PHI_RA1T_KEY = 'phi2km1p5_ra1t'

datasets = {N1280_GA6_KEY:{'model_name':'N1280 GA6 LAM Model'},
            KM4P4_RA1T_KEY:{'model_name':'SE Asia 4.4KM RA1-T '},
            KM1P5_INDO_RA1T_KEY:{'model_name':'Indonesia 1.5KM RA1-T'},
             KM1P5_MAL_RA1T_KEY:{'model_name':'Malaysia 1.5KM RA1-T'},
             KM1P5_PHI_RA1T_KEY:{'model_name':'Philipines 1.5KM RA1-T'},
           }

s3_base = '{server}/{bucket}/model_data/'.format(server=server_address,
                                                 bucket=bucket_name)
s3_local_base = os.path.join(os.sep,'s3',bucket_name, 'model_data')

for ds_name in datasets.keys():
    fname1 = 'SEA_{conf}_{fct}.nc'.format(conf=ds_name,
                                         fct=fcast_time)
    datasets[ds_name]['fname'] = fname1
    datasets[ds_name]['s3_url'] = os.path.join(s3_base, fname1)
    datasets[ds_name]['s3_local_path'] = os.path.join(s3_local_base, fname1)
    datasets[ds_name]['data'] = None


fname_key = 's3_local_path'
# This code downloads the files to the local disk before loading, rather than using the s3 FS mount. This is for using when
# S3 mount is unavailable, or for performance reasons.
do_download = True
if do_download:
    base_path_local = os.path.expanduser('~/SEA_data/model_data/')
    if not (os.path.isdir(base_path_local)):
        print('creating directory {0}'.format(base_path_local))
        os.makedirs(base_path_local)
        
    for ds_name in datasets.keys():
        datasets[ds_name]['local_path'] = os.path.join(base_path_local,
                                                       datasets[ds_name]['fname'])
           
    
    for ds_name in datasets:
        lib_sea.download_from_s3(datasets[ds_name]['s3_url'], datasets[ds_name]['local_path'])
        
    fname_key = 'local_path'
    

for ds_name in datasets:
    datasets[ds_name]['data'] = iris.load(datasets[ds_name][fname_key])





#set up datasets dictionary

plot_names = ['precipitation',
              'air_temperature',
              'wind_vectors',
                'wind_mslp',
               'wind_streams',
               'mslp',
               'cloud_fraction',

             ]

var_names = ['precipitation',
             'air_temperature',
             'wind_speed',
             'wind_vectors',
             'cloud_fraction',
             'mslp',
            ]

datasets[N1280_GA6_KEY]['var_lookup'] = {'precipitation':'precipitation_flux',
                                 'cloud_fraction': 'cloud_area_fraction_assuming_maximum_random_overlap',
                                 'air_temperature':'air_temperature',
                                 'x_wind':'x_wind',
                                 'y_wind':'y_wind',
                                 'mslp':'air_pressure_at_sea_level',
                                }
datasets[N1280_GA6_KEY]['units'] = {'precipitation':'kg-m-2-hour^-1',
                                 'cloud_fraction': None,
                                 'air_temperature':'celsius',
                                 'x_wind':'miles-hour^-1',
                                 'y_wind':'miles-hour^-1',
                                 'mslp':'hectopascals',
                                }
datasets[KM4P4_RA1T_KEY]['var_lookup'] = {'precipitation':'stratiform_rainfall_rate',
                                  'cloud_fraction': 'cloud_area_fraction_assuming_maximum_random_overlap',
                                  'air_temperature':'air_temperature',
                                  'x_wind':'x_wind',
                                  'y_wind':'y_wind',
                                  'mslp':'air_pressure_at_sea_level',
                                 }
datasets[KM4P4_RA1T_KEY]['units'] = {'precipitation':'kg-m-2-hour^-1',
                                 'cloud_fraction': None,
                                 'air_temperature':'celsius',
                                 'x_wind':'miles-hour^-1',
                                 'y_wind':'miles-hour^-1',
                                 'mslp':'hectopascals',
                                }

datasets[KM1P5_INDO_RA1T_KEY]['units'] = dict(datasets[KM4P4_RA1T_KEY]['units'])
datasets[KM1P5_MAL_RA1T_KEY]['units'] = dict(datasets[KM4P4_RA1T_KEY]['units'])
datasets[KM1P5_PHI_RA1T_KEY]['units'] = dict(datasets[KM4P4_RA1T_KEY]['units'])

datasets[KM1P5_INDO_RA1T_KEY]['var_lookup'] = dict(datasets[KM4P4_RA1T_KEY]['var_lookup'])
datasets[KM1P5_MAL_RA1T_KEY]['var_lookup'] = dict(datasets[KM4P4_RA1T_KEY]['var_lookup'])
datasets[KM1P5_PHI_RA1T_KEY]['var_lookup'] = dict(datasets[KM4P4_RA1T_KEY]['var_lookup'])




for ds_name in datasets:
    print('loading dataset {0}'.format(ds_name))
    for var1 in datasets[ds_name]['var_lookup']:
        print('    loading var {0}'.format(var1))
        datasets[ds_name][var1] = iris.load_cube(datasets[ds_name][fname_key], 
                                                 datasets[ds_name]['var_lookup'][var1])
        if datasets[ds_name]['units'][var1]:
            datasets[ds_name][var1].convert_units(datasets[ds_name]['units'][var1])


# process wind cubes to calculate wind speed
WIND_SPEED_NAME = 'wind_speed'
cube_pow = iris.analysis.maths.exponentiate
for ds_name in datasets:
    print('calculating wind speed for {0}'.format(ds_name))
    cube_x_wind = datasets[ds_name]['x_wind']
    cube_y_wind = datasets[ds_name]['y_wind']
    datasets[ds_name]['wind_speed'] = cube_pow( cube_pow(cube_x_wind, 2.0) +
                                                  cube_pow(cube_y_wind, 2.0),
                                                 0.5 )
    datasets[ds_name]['wind_speed'].rename(WIND_SPEED_NAME)


for ds_name in datasets:
    datasets[ds_name].update(lib_sea.calc_wind_vectors(datasets[ds_name]['x_wind'], 
                                               datasets[ds_name]['y_wind'],
                                               10))

# create regions
region_dict = {'indonesia': [-15.1, 1.0865, 99.875, 120.111],
               'malaysia': [-2.75, 10.7365, 95.25, 108.737],
               'phillipines': [3.1375, 21.349, 115.8, 131.987],
               'se_asia': [-18.0, 29.96, 90.0, 153.96],
              }

#Setup and display plots
plot_opts = lib_sea.create_colour_opts(plot_names)





init_time = 4
init_var = plot_names[0]
init_region = 'se_asia'
init_model_left = N1280_GA6_KEY # KM4P4_RA1T_KEY
init_model_right = KM4P4_RA1T_KEY # N1280_GA6_KEY


plot_obj_left = SEA_plot(datasets,
                         plot_opts,
                         'plot_sea_left',
                         init_var,
                         init_model_left,
                         init_region,
                         region_dict,
                        )

plot_obj_left.current_time = init_time
bokeh_img_left = plot_obj_left.create_plot()

plot_obj_right = SEA_plot(datasets,
                    plot_opts,
                    'plot_sea_right',
                    init_var,
                    init_model_right,
                    init_region,
                    region_dict,
                    )

plot_obj_right.current_time = init_time
bokeh_img_right = plot_obj_right.create_plot()

plots_row = bokeh.layouts.row(bokeh_img_left,
                              bokeh_img_right)

# set up bokeh widgets
def create_dropdown_opt_list(iterable1):
    return [(k1,k1) for k1 in iterable1]

model_var_list_desc = 'Attribute to visualise'

model_var_dd = \
    bokeh.models.widgets.Dropdown(label=model_var_list_desc,
                                  menu=create_dropdown_opt_list(plot_names),
                                  button_type='warning')
model_var_dd.on_change('value',plot_obj_left.on_var_change)
model_var_dd.on_change('value',plot_obj_right.on_var_change)

num_times = datasets[N1280_GA6_KEY]['precipitation'].shape[0]
for ds_name in datasets:
    num_times = min(num_times, datasets[ds_name]['precipitation'].shape[0])
    
    
data_time_slider = bokeh.models.widgets.Slider(start=0, 
                                               end=num_times, 
                                               value=init_time, 
                                               step=1, 
                                               title="Data time")
                                               
data_time_slider.on_change('value',plot_obj_right.on_data_time_change)
data_time_slider.on_change('value',plot_obj_left.on_data_time_change)

region_desc = 'Region'

region_menu_list = create_dropdown_opt_list(region_dict.keys())
region_dd = bokeh.models.widgets.Dropdown(menu=region_menu_list, 
                                          label=region_desc,
                                          button_type='warning')
region_dd.on_change('value', plot_obj_right.on_region_change)
region_dd.on_change('value', plot_obj_left.on_region_change)

dataset_menu_list = create_dropdown_opt_list(datasets.keys())
left_model_desc = 'Left display'

left_model_dd = bokeh.models.widgets.Dropdown(menu=dataset_menu_list,
                                               label=left_model_desc,
                                               button_type='warning')
left_model_dd.on_change('value', plot_obj_left.on_config_change,)


right_model_desc = 'Right display'
right_model_dd = bokeh.models.widgets.Dropdown(menu=dataset_menu_list, 
                                             label=right_model_desc,
                                             button_type='warning')
right_model_dd.on_change('value', plot_obj_right.on_config_change)

# layout widgets
param_row = bokeh.layouts.row(model_var_dd, region_dd)
slider_row = bokeh.layouts.row(data_time_slider)
config_row = bokeh.layouts.row(left_model_dd, right_model_dd)

main_layout = bokeh.layouts.column(param_row, 
                                   slider_row,
                                   config_row,
                                   plots_row,
                                   )

try:
    bokeh_mode = os.environ['BOKEH_MODE']
except:
    bokeh_mode = 'server'    
    
if bokeh_mode == 'server':
    bokeh.plotting.curdoc().add_root(main_layout)
elif bokeh_mode == 'cli':
    bokeh.io.show(main_layout)
    
# Share axes between plots to enable linked zooming and panning
#plot_obj_left.share_axes([plot_obj_right.current_axes])


