### SE Asia Model and GPM IMERG Matplotlib example Bokeh app script

# This script demonstrates creating plots of model rainfall data and GPM IMERG
#  data for SE Asia using the Matplotlib plotting library to provide images to
#  a Bokeh Server App.

## Setup notebook
# Do module imports

import os
import time
import sys
import datetime as dt
import textwrap
import numpy as np
import matplotlib as mpl
mpl.use('agg')

import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import iris
import bokeh.io
import bokeh.layouts 
import bokeh.models.widgets
import bokeh.plotting

iris.FUTURE.netcdf_promote = True

import lib_sea

## Extract
# Extract data from S3. The data can either be downloaded in full before 
#  loading, or downloaded on demand using the /s3 filemount. This is 
#  controlled by the do_download flag.

bucket_name = 'stephen-sea-public-london'
server_address = 'https://s3.eu-west-2.amazonaws.com'

fcast_time = '20180109T0000Z'
fcast_time_obj =  dt.datetime.strptime(fcast_time, '%Y%m%dT%H%MZ')

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
                                    dt.timedelta(days = dd)).strftime('%Y%m%d'))
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
    return math.floor(value/3.)*3

for ds_name in datasets:
    
    if 'gpm_imerg' not in ds_name:
        continue

    temp_cube_list = iris.cube.CubeList()

    for time in datasets[ds_name]['data'].keys():
        raw_cube = datasets[ds_name]['data'][time]
        iris.coord_categorisation.add_categorised_coord(raw_cube, 'agg_time', 'time', conv_func, 
                                                        units=iris.unit.Unit('hours since 1970-01-01', 
                                                                             calendar='gregorian'))
        accum_cube = raw_cube.aggregated_by( ['agg_time'], iris.analysis.SUM)
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

# The following cell contains the class SEA_plot, which contains code to create and display all the various plots possible 
#  with this notebook. The inspiration for this structure is the model-view-controller (MVC) paradigm  
#  (https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93controller). This class is both the view and the controller, as  
#  it creates the plots and handles events.  The datasets dictionary is currently the model element in this notebook.
# There are main motivations for choosing this structure. The first is to facilitate the future expansion of this plotting  
#  framework. Hopefully new datasets or plot types can be added by merely defining new plotting functions or adding elements  
#  to the datasets dictionary. Secondly, this structure is hopefully sufficiently flexible to be able to change the libraries  
#  or functions used for the actual plotting, without having to start from scratch.
# This is currently an imperfect interpretation of the MVC paradigm. In future development, the SEA_plot class should have  
#  the event handlers split off into a controller class. The datasets dictionary should probably also be given a more  
#  object-oriented (OO) treament in the future to make inclusion of further datasets easier. 

class SEA_plot(object):
    '''
    Main plot class. The plotting function is create_plot().
    '''
    TITLE_TEXT_WIDTH = 40

    def __init__(self, dataset, po1, figname, plot_type, conf1, reg1):
        '''
        Initialisation function for SEA_plot class
        '''
        self.main_plot = None
        self.current_time = 0
        self.plot_options = po1
        self.dataset = dataset
        self.figure_name = figname
        self.current_type = plot_type
        self.set_config(conf1)
        self.current_region = reg1
        self.data_bounds = region_dict[self.current_region]
        self.use_mpl_title = False
        self.show_axis_ticks = False
        self.show_colorbar = False
        self.setup_plot_funcs()
        self.current_title = ''

    def setup_plot_funcs(self):
    
        '''
        Set up dictionary of plot functions. This is used by the main create_plot() function 
        to call the plotting function relevant to the specific variable being plotted. There
        is also a second dictionary which is by the update_plot() function, which does the minimum
        amount of work to update the plot, and is used for some option changes, mainly a change 
        in the forecast time selected.
        '''
        self.plot_funcs = {'precipitation' : self.plot_precip}
    
    def set_config(self, new_config):
        self.current_config = new_config
        self.plot_description = self.dataset[self.current_config]['data_type_name']
    
    def update_coords(self, data_cube):
    
        '''
        Update the latitude and longitude coordinates for the data.
        '''
        
        self.coords_lat = data_cube.coords('latitude')[0].points
        self.coords_long = data_cube.coords('longitude')[0].points
        
    def plot_precip(self):
    
        '''
        Function for creating precipitation plots, called by create_plot when 
        precipitation is the selected plot type.
        '''
        
        data_cube = self.dataset[self.current_config][self.current_type]

        self.update_coords(data_cube)
        self.current_axes.coastlines(resolution='110m')
        self.main_plot = self.current_axes.pcolormesh(self.coords_long, 
                                                      self.coords_lat, 
                                                      data_cube[self.current_time].data, 
                                                      cmap=self.plot_options[self.current_type]['cmap'],
                                                      norm=self.plot_options[self.current_type]['norm'],
                                                      transform=ccrs.PlateCarree())
        self.update_title(data_cube)
 
    def update_title(self, current_cube):
        '''
        Update plot title.
        '''
        datestr1 = lib_sea.get_time_str(current_cube.dim_coords[0].points[self.current_time])
        
        str1 = '{plot_desc} {var_name} at {fcst_time}'.format(var_name=self.current_type,
                                                              fcst_time=datestr1,
                                                              plot_desc=self.plot_description,
                                                             )
        self.current_title = '\n'.join(textwrap.wrap(str1, 
                                                     SEA_plot.TITLE_TEXT_WIDTH)) 
        
    def create_plot(self):
    
        '''
        Main plotting function. Generic elements of the plot are created here, and then the plotting
        function for the specific variable is called using the self.plot_funcs dictionary.
        '''
        
        self.create_matplotlib_fig()
        self.create_bokeh_img_plot_from_fig()
        
        return self.bokeh_figure

    def create_matplotlib_fig(self):
    
        ''' 
        A method for creating a matplotlib data plot.
        '''
        
        self.current_figure = plt.figure(self.figure_name, figsize=(4.0, 3.0))
        self.current_figure.clf()
        self.current_axes = self.current_figure.add_subplot(111, projection=ccrs.PlateCarree())
        self.current_axes.set_position([0, 0, 1, 1])

        self.plot_funcs[self.current_type]()
        if self.use_mpl_title:
            self.current_axes.set_title(self.current_title)
        self.current_axes.set_xlim(90, 154)
        self.current_axes.set_ylim(-18, 30)
        self.current_axes.xaxis.set_visible(self.show_axis_ticks)
        self.current_axes.yaxis.set_visible(self.show_axis_ticks)
        if self.show_colorbar:
            self.current_figure.colorbar(self.main_plot,
                                         orientation='horizontal')

        self.current_figure.canvas.draw()
        
    def create_bokeh_img_plot_from_fig(self):
    
        '''
        A method to create a Bokeh figure displaying a matplotlib-generated
         image.
        '''
        
        self.current_img_array = lib_sea.get_image_array_from_figure(self.current_figure)
        
        # Set figure navigation limits
        x_limits = bokeh.models.Range1d(90, 154, bounds = (90, 154))
        y_limits = bokeh.models.Range1d(-18, 30, bounds = (-18, 30))
        
        # Initialize figure        
        self.bokeh_figure = bokeh.plotting.figure(plot_width=800, 
                                                  plot_height=600, 
                                                  x_range = x_limits,
                                                  y_range = y_limits, 
                                                  tools = 'pan,wheel_zoom,reset')
                                                  
        latitude_range = 48
        longitude_range = 64
        self.bokeh_image = self.bokeh_figure.image_rgba(image=[self.current_img_array], 
                                                        x=[90], 
                                                        y=[-18], 
                                                        dw=[longitude_range], 
                                                        dh=[latitude_range])
        self.bokeh_figure.title.text = self.current_title
        
        self.bokeh_img_ds = self.bokeh_image.data_source
        
    def update_bokeh_img_plot_from_fig(self):
    
        '''
        
        '''
        
        self.current_img_array = lib_sea.get_image_array_from_figure(self.current_figure)
        self.bokeh_figure.title.text = self.current_title
        self.bokeh_img_ds.data[u'image'] = [self.current_img_array]  
        
    def on_data_time_change(self, attr1, old_val, new_val):
    
        '''
        Event handler for a change in the selected forecast data time.
        '''
        
        print('selected new time {0}'.format(new_val))
        self.current_time = int(new_val/3)
        self.create_matplotlib_fig()
        self.update_bokeh_img_plot_from_fig()

    def on_date_slider_change(self, attr1, old_val, new_val):
        '''
        Event handler for a change in the selected forecast data date.
        '''
        print('selected new date {0}'.format(new_val))
        self.current_time = new_val.strftime('%Y%m%d') + self.current_time[-4:]
        self.create_matplotlib_fig()
        self.update_bokeh_img_plot_from_fig()
    
    def on_hour_slider_change(self, attr1, old_val, new_val):
        '''
        Event handler for a change in the selected forecast data date.
        '''
        print('selected new date {0}'.format(new_val))
        self.current_time = self.current_time[:-4] + '{:02d}00'.format(new_val)
        self.create_matplotlib_fig()
        self.update_bokeh_img_plot_from_fig()
        
    def on_type_change(self, attr1, old_val, new_val):
    
        '''
        Event handler for a change in the selected plot type.
        '''
        
        print('selected new var {0}'.format(new_val))
        self.current_type = new_val
        self.create_matplotlib_fig()
        self.update_bokeh_img_plot_from_fig()
    
    def on_config_change(self, attr1, old_val, new_val):
        '''
        Event handler for a change in the selected model configuration output.
        '''
        print('selected new config {0}'.format(new_val))
        self.set_config(new_val)
        self.create_matplotlib_fig()
        self.update_bokeh_img_plot_from_fig()

    def on_imerg_change(self, attr1, old_val, new_val):
        '''
        Event handler for a change in the selected model configuration output.
        '''
        imerg_list = [ds_name for ds_name in datasets.keys() 
                      if 'imerg' in ds_name]
        print('selected new config {0}'.format(imerg_list[new_val]))
        self.set_config(imerg_list[new_val])
        self.create_matplotlib_fig()
        self.update_bokeh_img_plot_from_fig()
        
    def link_axes_to_other_plot(self, other_plot):
        try:
            self.bokeh_figure.x_range = other_plot.bokeh_figure.x_range
            self.bokeh_figure.y_range = other_plot.bokeh_figure.y_range
        except:
            print('bokeh plot linking failed.')        
               
# Set the initial values to be plotted
init_time = 12
init_var = 'precipitation'
init_region = 'se_asia'
init_model_left = KM4P4_RA1T_KEY
init_model_right = GPM_IMERG_EARLY_KEY


## Display plots

# Create a plot object for the left model display
plot_obj_left = SEA_plot(datasets,
                         plot_opts,
                         'plot_sea_left',
                         init_var,
                         init_model_left,
                         init_region,
                         )

plot_obj_left.current_time = init_time
bokeh_img_left = plot_obj_left.create_plot()

# Create a plot object for the right model display
plot_obj_right = SEA_plot(datasets,
                         plot_opts,
                         'plot_sea_right',
                         init_var,
                         init_model_right,
                         init_region,
                         )

plot_obj_right.current_time = init_time
bokeh_img_right = plot_obj_right.create_plot()

plot_obj_right.link_axes_to_other_plot(plot_obj_left)

plots_row = bokeh.layouts.row(bokeh_img_left, bokeh_img_right)

# Set up bokeh widgets
def create_dropdown_opt_list(iterable1):
    return [(k1,k1) for k1 in iterable1]

num_times = 3*datasets[GPM_IMERG_LATE_KEY]['precipitation'].shape[0]
    
data_time_slider = bokeh.models.widgets.Slider(start=0, 
                                               end=num_times, 
                                               value=init_time, 
                                               step=3, 
                                               title="Data time",
                                               width = 800)
                                               
data_time_slider.on_change('value', plot_obj_right.on_data_time_change)
data_time_slider.on_change('value', plot_obj_left.on_data_time_change)

'''start_date = fcast_time_obj.date()
end_date = (start_date + dt.timedelta(days = 3))
value_date = dt.date.strptime(init_time[:8], '%Y%m%d')

date_slider = bokeh.models.widgets.sliders.DateSlider(start = start_date,
                                                      end = end_date,
                                                      value = value_date,
                                                      step = 86400000, 
                                                      title = 'Select hour')

date_slider.on_change('value', plot_obj_left.on_date_slider_change)
date_slider.on_change('value', plot_obj_right.on_date_slider_change)

hour_slider = bokeh.models.widgets.sliders.Slider(start = 0,
                                                  end = 21,
                                                  value = 12,
                                                  step = 3,
                                                  title = 'Select hour')

hour_slider.on_change('value', plot_obj_left.on_hour_slider_change)
hour_slider.on_change('value', plot_obj_right.on_hour_slider_change)'''

model_menu_list = create_dropdown_opt_list([ds_name for ds_name in datasets.keys() 
                                            if 'imerg' not in ds_name])
gpm_imerg_menu_list = create_dropdown_opt_list([ds_name for ds_name in datasets.keys() 
                                                if 'imerg' in ds_name])

model_dd_desc = 'Model display'
model_dd = bokeh.models.widgets.Dropdown(menu = model_menu_list,
                                         label = model_dd_desc,
                                         button_type = 'warning',
                                         width = 800)
model_dd.on_change('value', plot_obj_left.on_config_change,)

imerg_rbg = bokeh.models.widgets.RadioButtonGroup(labels = [ds_name for ds_name 
                                                           in datasets.keys() 
                                                           if 'imerg' in ds_name], 
                                                 button_type = 'warning',
                                                 width = 800)
imerg_rbg.on_change('active', plot_obj_right.on_imerg_change)
                                
                              
# Set layout for widgets
slider_row = bokeh.layouts.row(data_time_slider)
config_row = bokeh.layouts.row(model_dd, imerg_rbg, width = 1600)

main_layout = bokeh.layouts.column(slider_row,
                                   config_row,
                                   plots_row
                                   )

try:
    bokeh_mode = os.environ['BOKEH_MODE']
except:
    bokeh_mode = 'server'    
    
if bokeh_mode == 'server':
    bokeh.plotting.curdoc().add_root(main_layout)
elif bokeh_mode == 'cli':
    bokeh.io.show(main_layout)
    
bokeh.plotting.curdoc().title = 'Model rainfall vs GPM app'    
