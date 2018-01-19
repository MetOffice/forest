import os
import time
import sys
import urllib.request
import textwrap
import functools
import threading

import numpy
import iris

import bokeh.io
import bokeh.layouts 
import bokeh.models.widgets
import bokeh.plotting


import matplotlib
matplotlib.use('agg')

import lib_sea
import sea_plot
import sea_data
import sea_control

iris.FUTURE.netcdf_promote = True

try:
    get_ipython
    is_notbook = True
except:
    is_notebook = False

def add_main_plot(main_layout, bokeh_doc):
    print('finished creating, executing document add callback')

    try:
        bokeh_mode = os.environ['BOKEH_MODE']
    except:
        bokeh_mode = 'server'    
        
    if bokeh_mode == 'server':
        bokeh_doc.add_root(main_layout)
    elif bokeh_mode == 'cli':
        bokeh.io.show(main_layout)


# Extract and Load
bucket_name = 'stephen-sea-public-london'
server_address = 'https://s3.eu-west-2.amazonaws.com'


fcast_dt_list, fcast_dt_str_list = lib_sea.get_model_run_times(7)

fcast_time = '20180110T0000Z'

#fcast_time = fcast_dt_str_list[-2]

# Setup datasets. Data is not loaded until requested for plotting.
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
for ds_name in datasets.keys():
    datasets[ds_name]['var_lookup'] = sea_data.VAR_LOOKUP_RA1T

datasets[N1280_GA6_KEY]['var_lookup'] = sea_data.VAR_LOOKUP_GA6    

s3_base = '{server}/{bucket}/model_data/'.format(server=server_address,
                                                 bucket=bucket_name)
s3_local_base = os.path.join(os.sep,'s3',bucket_name, 'model_data')
base_path_local = os.path.expanduser('~/SEA_data/model_data/')
use_s3_mount = False
do_download = True


for ds_name in datasets.keys():
    fname1 = 'SEA_{conf}_{fct}.nc'.format(conf=ds_name, fct=fcast_time)
    datasets[ds_name]['data'] = sea_data.SEA_dataset(ds_name, 
                                                     fname1,
                                                     s3_base,
                                                     s3_local_base,
                                                     use_s3_mount,
                                                     base_path_local,
                                                     do_download,
                                                     datasets[ds_name]['var_lookup'],
                                                     )


#set up datasets dictionary

plot_names = ['precipitation',
              'air_temperature',
              'wind_vectors',
                'wind_mslp',
               'wind_streams',
               'mslp',
               'cloud_fraction',
               #'blank',
               ]



# create regions
region_dict = {'indonesia': [-15.1, 1.0865, 99.875, 120.111],
               'malaysia': [-2.75, 10.7365, 95.25, 108.737],
               'phillipines': [3.1375, 21.349, 115.8, 131.987],
               'se_asia': [-18.0, 29.96, 90.0, 153.96],
              }

#Setup and display plots
plot_opts = lib_sea.create_colour_opts(plot_names)





init_time = 4
init_var = plot_names[0] #blank
init_region = 'se_asia'
init_model_left = N1280_GA6_KEY # KM4P4_RA1T_KEY
init_model_right = KM4P4_RA1T_KEY # N1280_GA6_KEY

#Set up plots
plot_obj_left = sea_plot.SEA_plot(datasets,
                        plot_opts,
                        'plot_sea_left',
                        init_var,
                        init_model_left,
                        init_region,
                        region_dict,
                        sea_data.UNIT_DICT,
                        )

plot_obj_left.current_time = init_time
bokeh_img_left = plot_obj_left.create_plot()
stats_left = plot_obj_left.create_stats_widget()

plot_obj_right = sea_plot.SEA_plot(datasets,
                    plot_opts,
                    'plot_sea_right',
                    init_var,
                    init_model_right,
                    init_region,
                    region_dict,
                    sea_data.UNIT_DICT,
                    )


plot_obj_right.current_time = init_time
bokeh_img_right = plot_obj_right.create_plot()
stats_right = plot_obj_right.create_stats_widget()

plots_row = bokeh.layouts.row(bokeh_img_left,
                            bokeh_img_right)

plot_obj_right.link_axes_to_other_plot(plot_obj_left)



num_times = datasets[N1280_GA6_KEY]['data'].get_data('precipitation').shape[0]
for ds_name in datasets:
    num_times = min(num_times, datasets[ds_name]['data'].get_data('precipitation').shape[0])
bokeh_doc = bokeh.plotting.curdoc()

# Set up GUI controller class
control1 = sea_control.SEA_controller(init_time,
                                      num_times,
                                      datasets,
                                      plot_names,
                                      [plot_obj_left, plot_obj_right],
                                      [bokeh_img_left, bokeh_img_right],
                                      [stats_left, stats_right],
                                      region_dict,
                                      bokeh_doc,
                                      )

add_main_plot(control1.main_layout, bokeh_doc)

bokeh_doc.title = 'Two model comparison - Lazy loading'    
