### SE Asia Sim. Im. and Himawari-8 Matplotlib example notebook

# This script demostrates creating plots of simulated satellite imagery and 
#  Himawari-8 imagery for SE Asia using the Matplotlib plotting library to
#  provide images to a Bokeh Server App.

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

SIMIM_KEY = 'simim'
HIMAWARI8_KEY = 'himawari-8'

# The datasets dictionary is the main container for the sim. im./Himawari-8
#  data and associated meta data. It is stored as a dictionary of dictionaries.
# The first layer of indexing selects the data type, for example Simulated 
#  Imagery or Himawari-8 imagery. Each of these will be populated with a cube
#  or data image array for each of the available wavelengths as well as 
#  asociated metadata such as file paths etc.

datasets = {SIMIM_KEY:{'data_type_name': 'Simulated Imagery'},
            HIMAWARI8_KEY:{'data_type_name': 'Himawari-8 Imagery'},
           }

# Himawari-8 imagery dict population

s3_base_str = '{server}/{bucket}/himawari-8/'
s3_base = s3_base_str.format(server = server_address, bucket = bucket_name)
s3_local_base = os.path.join(os.sep, 's3', bucket_name, 'himawari-8')

fnames_list = ['{im}_{datetime}.jpg'.format(im = im_type, datetime = (fcast_time_obj + dt.timedelta(hours = vt)).strftime('%Y%m%d%H%M')) 
               for vt in range(12, 39, 3) for im_type in ['LIIN50', 'LVIN10', 'LWIN11']]
datasets['himawari-8']['fnames_list'] = fnames_list
datasets['himawari-8']['s3_url_list'] = [os.path.join(s3_base, fname) 
                                         for fname in fnames_list]
datasets['himawari-8']['s3_local_path_list'] = [os.path.join(s3_local_base, fname)
                                                for fname in fnames_list]

# Simulated imagery dict population

s3_base_str = '{server}/{bucket}/simim/'
s3_base = s3_base_str.format(server = server_address, bucket = bucket_name)
s3_local_base = os.path.join(os.sep, 's3', bucket_name, 'simim')

fnames_list = ['sea4-{it}_HIM8_{date}_s4{run}_T{time}.nc'.format(it=im_type, date=fcast_time[:8], run=fcast_time[9:11], time = vt) 
               for vt in range(12, 39, 3) for im_type in ['simbt', 'simvis']]
datasets['simim']['fnames_list'] = fnames_list
datasets['simim']['s3_url_list'] = [os.path.join(s3_base, fname)
                                    for fname in fnames_list]
datasets['simim']['s3_local_path_list'] = [os.path.join(s3_local_base, fname)
                                           for fname in fnames_list]

fname_key = 's3_local_path'

# Download the data from S3

# This code downloads the files to the local disk before loading, rather than 
#  using the s3 FS mount. This is for using when S3 mount is unavailable, or 
#  for performance reasons.
do_download = True
use_jh_paths = True
base_dir = os.path.expanduser('~/SEA_data/')
for dtype in datasets.keys():
    if do_download:
        if use_jh_paths:
            base_path_local = os.path.join(base_dir,dtype)
        else:
            base_path_local = '/usr/local/share/notebooks/{data_type}/'.format(data_type=dtype)
        if not (os.path.isdir(base_path_local)):
            print('creating directory {0}'.format(base_path_local))
            os.makedirs(base_path_local)
            
        datasets[dtype]['local_paths_list'] = [os.path.join(base_path_local, file_name)
                                               for file_name in datasets[dtype]['fnames_list']]

        for s3_url, local_path in zip(datasets[dtype]['s3_url_list'], 
                                      datasets[dtype]['local_paths_list']):
            try:
                lib_sea.download_from_s3(s3_url, local_path)
            except:
                print("    Warning: file not downloaded successfully:", s3_url)

fname_key = 'local_path'


##   Load
# Load data using iris. 

# Write Simulated Imagery data to  a dictionary indexed by time string

datasets['simim']['data'] = {'W': {}, 'I': {}, 'V':{}}

for file_name in datasets['simim']['local_paths_list']:
    
    cube_time_td = dt.timedelta(hours = int(file_name[-5:-3]))
    cube_time_str = ((fcast_time_obj + cube_time_td).strftime('%Y%m%d%H%M'))
    if os.path.isfile(file_name):
        cube_list = iris.load(file_name)
    else: 
        continue
    if 'simbt' in file_name:
        datasets['simim']['data']['W'].update({cube_time_str: cube_list[0]})
        datasets['simim']['data']['I'].update({cube_time_str: cube_list[2]})
    elif 'simvis' in file_name:
        datasets['simim']['data']['V'].update({cube_time_str: cube_list[0]})
        
# Write Himawari-8 image data to Numpy arrays in a dictionary indexed by time
#  string

datasets['himawari-8']['data']={'W': {}, 'I': {}, 'V':{}}

for im_type in ['W', 'V', 'I']:
    
    im_array_dict = {}
    
    for file_name in datasets['himawari-8']['local_paths_list']:

        if 'L' + im_type + 'IN' in file_name:
            try:
                im_array_dict.update({file_name[-16:-4]: plt.imread(file_name)})
            except:
                pass
            
    datasets['himawari-8']['data'][im_type].update(im_array_dict)
    

## Setup plots

from matplotlib.colors import LinearSegmentedColormap

def loadct_SPSgreyscale():

    ''' A function to make a custom greyscale colormap to match Tigger plots.
         In a 256 colour setup I need to remove 4 colours (black and the three
         darkest greys) from  the black end of the greyscale and one (white) 
         from the white end.
    '''
    
    n = 256.0
    color_min = 1.0
    color_max = 252.0
    cdict_min = color_min / n
    cdict_max = color_max / n
    ncolours=(n-color_min)-(n-color_max)
    cdict = {'red': [(0.0, cdict_max, cdict_max),
                     (1.0, cdict_min, cdict_min)],
             'green': [(0.0, cdict_max, cdict_max),
                     (1.0, cdict_min, cdict_min)], 
             'blue': [(0.0, cdict_max, cdict_max),
                     (1.0, cdict_min, cdict_min)]}
    SPSgreyscale = LinearSegmentedColormap('SPSgreyscale', cdict, int(ncolours))
    plt.register_cmap(name='SPSgreyscale', cmap=SPSgreyscale)

loadct_SPSgreyscale()

plot_options = {'V': {'norm': mpl.colors.Normalize(0, 1), 
                      'cmap': 'binary_r'},
                'W': {'norm': mpl.colors.Normalize(198, 308), 
                      'cmap': 'SPSgreyscale'},
                'I': {'norm': mpl.colors.Normalize(198, 308), 
                      'cmap': 'SPSgreyscale'}}

# The following cell contains the class SEA_plot, which contains code to create
#  and display all the various plots possible with this notebook. The 
#  inspiration for this structure is the model-view-controller (MVC) paradigm 
#  (https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93controller). This 
#  class is both the view and the controller, as it creates the plots and 
#  handles events.  The datasets dictionary is currently the model element in 
#  this script.
# There are main motivations for choosing this structure. The first is to 
#  facilitate the future expansion of this plotting framework. Hopefully new 
#  datasets or plot types can be added by merely defining new plotting functions 
#  or adding elements to the datasets dictionary. Secondly, this structure is
#  hopefully sufficiently flexible to be able to change the libraries or 
#  functions used for the actual plotting, without having to start from scratch.
# This is currently an imperfect interpretation of the MVC paradigm. In future 
#  development, the SEA_plot class should have the event handlers split off 
#  into a controller class. The datasets dictionary should probably also be 
#  given a more object-oriented (OO) treament in the future to make inclusion 
#  of further datasets easier. 

class SEA_plot(object):
    '''
    Main plot class. The plotting function is create_plot().
    '''
    TITLE_TEXT_WIDTH = 40
    def __init__(self, dataset, po1, figname, plot_type, time, conf):
        '''
        Initialisation function for SEA_plot class
        '''
        self.main_plot = None
        self.plot_options = po1
        self.dataset = dataset
        self.figure_name = figname
        self.current_type = plot_type
        self.current_time = time
        self.set_config(conf)
        self.use_mpl_title = False
        self.show_axis_ticks = False
        self.show_colorbar = False
        self.setup_plot_funcs()
    
    def setup_plot_funcs(self):
    
        '''
        Set up dictionary of plot functions. This is used by the main 
        create_plot() function to call the plotting function relevant to the 
        specific variable being plotted. There is also a second dictionary 
        which is by the update_plot() function, which does the minimum amount 
        of work to update the plot, and is used for some option changes, mainly 
        a change in the forecast time selected.
        '''
        
        self.plot_funcs = {'simim': self.plot_simim,
                           'himawari-8': self.plot_him8,
                          }
        self.update_funcs = {'simim': self.update_simim,
                             'himawari-8': self.update_him8,
                            }
                            
    def set_config(self, new_config):
        self.current_config = new_config
        self.plot_description = self.dataset[self.current_config]['data_type_name']
    
          
    def update_him8(self):
    
        '''
        Update function for himawari-8 image plots, called by update_plot()  
        when cloud fraction is the selected plot type.
        '''
        
        '''
        cloud_cube = datasets['himawari-8'][self.current_type]
        array_for_update = cloud_cube[self.current_time].data[:-1,:-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title()'''
    
    def plot_him8(self):
    
        '''
        Function for creating himawari-8 image plots, called by create_plot()  
        when cloud fraction is the selected plot type.
        '''
        
        him8_image = datasets['himawari-8']['data'][self.current_type][self.current_time]
        self.current_axes.imshow(him8_image, extent=(90, 154, -18, 30), 
                                 origin='upper')
        
        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical', 
                                                            'coastline', 
                                                            '50m', 
                                                            edgecolor='g', 
                                                            alpha = 0.5,
                                                            facecolor = 'none')
        
        self.current_axes.add_feature(coastline_50m)  
        self.current_axes.set_extent((90, 154, -18, 30))

        self.update_title()
 
    def update_simim(self):
    
        '''
        Update function for himawari-8 image plots, called by update_plot() when 
        cloud fraction is the selected plot type.
        '''
        
        '''simim_cube = datasets['simim']['data'][self.current_type][self.current_time]
        array_for_update = cloud_cube.data[:-1,:-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title()'''
    
    def plot_simim(self):
    
        '''
        Function for creating himawari-8 image plots, called by create_plot() 
        when cloud fraction is the selected plot type.
        '''
        
        simim_cube = datasets['simim']['data'][self.current_type][self.current_time]
        lats = simim_cube.coords('grid_latitude')[0].points
        lons = simim_cube.coords('grid_longitude')[0].points
        self.main_plot = self.current_axes.pcolormesh(lons, 
                                                      lats, 
                                                      simim_cube.data,
                                                      cmap=plot_options[self.current_type]['cmap'],
                                                      norm=plot_options[self.current_type]['norm']
                                                      )

        # Add coastlines to the map created by contourf
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical', 
                                                            'coastline', 
                                                            '50m', 
                                                            edgecolor='g', 
                                                            alpha = 0.5,
                                                            facecolor = 'none')
        
        self.current_axes.add_feature(coastline_50m)  
        self.current_axes.set_extent((90, 154, -18, 30))
        self.update_title()

    def update_title(self):
    
        '''
        Update plot title.
        '''
        
        str1 = '{plot_desc} {var_name} at {time}'.format(plot_desc=self.plot_description,
                                                         var_name=self.current_type,
                                                         time=self.current_time,
                                                         )
        self.current_title = '\n'.join(textwrap.wrap(str1, 
                                                     SEA_plot.TITLE_TEXT_WIDTH)) 
        self.current_title = str1
        
    def create_plot(self):
    
        '''
        Main plotting function. Generic elements of the plot are created here, and then the plotting
        function for the specific variable is called using the self.plot_funcs dictionary.
        '''
        
        self.create_matplotlib_fig()
        self.create_bokeh_img_plot_from_fig()
        
        return self.bokeh_figure

    def update_plot(self):
    
        '''
        Main plot update function. Generic elements of the plot are updated here where possible, and then
        the plot update function for the specific variable is called using the self.plot_funcs dictionary.
        '''
        
        self.update_funcs[self.current_type]()
        if self.use_mpl_title:
            self.current_axes.set_title(self.current_title)
        self.current_figure.canvas.draw_idle()
        
        self.update_bokeh_img_plot_from_fig()
    
    def create_matplotlib_fig(self):
    
        ''' 
        A method for creating a matplotlib data plot.
        '''
        
        self.current_figure = plt.figure(self.figure_name, figsize=(8.0, 6.0))
        self.current_figure.clf()
        self.current_axes = self.current_figure.add_subplot(111, projection=ccrs.PlateCarree())
        self.current_axes.set_position([0, 0, 1, 1])

        self.plot_funcs[self.current_config]()
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
        print('size of image array is {0}'.format(self.current_img_array.shape))
        
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
        
    def share_axes(self, axes_list):
    
        '''
        Sets the axes of this plot to be shared with the other axes in the supplied list.
        '''
        
        self.current_axes.get_shared_x_axes().join(self.current_axes, *axes_list)
        self.current_axes.get_shared_y_axes().join(self.current_axes, *axes_list)
        
    def on_data_time_change(self, attr1, old_val, new_val):
        '''
        Event handler for a change in the selected forecast data time.
        '''
        print('selected new time {0}'.format(new_val))
        self.current_time = new_val[:-3]
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
        
    def link_axes_to_other_plot(self, other_plot):
        try:
            self.bokeh_figure.x_range = other_plot.bokeh_figure.x_range
            self.bokeh_figure.y_range = other_plot.bokeh_figure.y_range
        except:
            print('bokeh plot linking failed.')        
        
# Set the initial values to be plotted
init_time = '201801091200'
init_var = 'I'


## Display plots

# Create a plot object for the left model display
plot_obj_left = SEA_plot(datasets,
                         None,
                         'plot_sea_left',
                         init_var,
                         init_time,
                         'simim'
                         )

plot_obj_left.current_time = init_time
bokeh_img_left = plot_obj_left.create_plot()

# Create a plot object for the right model display
plot_obj_right = SEA_plot(datasets,
                          None,
                          'plot_sea_right',
                          init_var,
                          init_time,
                          'himawari-8'
                          )

plot_obj_right.current_time = init_time
bokeh_img_right = plot_obj_right.create_plot()

plot_obj_right.link_axes_to_other_plot(plot_obj_left)

plots_row = bokeh.layouts.row(bokeh_img_left,
                              bokeh_img_right)


# Set up bokeh widgets
def create_dropdown_opt_list(iterable1):
    return [(k1,k1) for k1 in iterable1]

wavelengths_list = ['W', 'I']

wavelength_dd = \
    bokeh.models.widgets.Dropdown(label = 'Wavelength',
                                  menu = create_dropdown_opt_list(wavelengths_list),
                                  button_type = 'warning')

wavelength_dd.on_change('value', plot_obj_left.on_type_change)
wavelength_dd.on_change('value', plot_obj_right.on_type_change)
                                
time_list = sorted([time_str + 'UTC' for time_str in datasets['simim']['data']['I'].keys() 
                    if time_str in datasets['himawari-8']['data']['I'].keys()])

data_time_dd = \
    bokeh.models.widgets.Dropdown(label = 'Time',
                                  menu = create_dropdown_opt_list(time_list),
                                  button_type = 'warning')
                                               
data_time_dd.on_change('value', plot_obj_right.on_data_time_change)
data_time_dd.on_change('value', plot_obj_left.on_data_time_change)

start_date = fcast_time_obj.date()
end_date = (start_date + dt.timedelta(days = 1))
value_date = dt.datetime.strptime(init_time[:8], '%Y%m%d').date()

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
hour_slider.on_change('value', plot_obj_right.on_hour_slider_change)

# Set layout for widgets
dd_row = bokeh.layouts.row(wavelength_dd, data_time_dd)
slider_row = bokeh.layouts.row(date_slider, hour_slider)

main_layout = bokeh.layouts.column(dd_row, 
                                   slider_row,
                                   plots_row)

try:
    bokeh_mode = os.environ['BOKEH_MODE']
except:
    bokeh_mode = 'server'    
    
if bokeh_mode == 'server':
    bokeh.plotting.curdoc().add_root(main_layout)
elif bokeh_mode == 'cli':
    bokeh.io.show(main_layout)

bokeh.plotting.curdoc().title = 'Model simulated imagery vs Himawari-8'    
