### SE Asia Sim. Im. and Himawari-8 Matplotlib example notebook

# This script demostrates creating plots of simulated satellite imagery and 
#  Himawari-8 imagery for SE Asia using the Matplotlib plotting library to
#  provide images to a Bokeh Server App.

## Setup notebook
# Do module imports

import os

import datetime

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot
import iris

import bokeh.models.widgets
import bokeh.plotting

iris.FUTURE.netcdf_promote = True

import simim_sat_control
import simim_sat_plot
import forest.util

## Extract
# Extract data from S3. The data can either be downloaded in full before 
#  loading, or downloaded on demand using the /s3 filemount. This is 
#  controlled by the do_download flag.

bokeh_id = __name__
bucket_name = 'stephen-sea-public-london'
server_address = 'https://s3.eu-west-2.amazonaws.com'

fcast_time = '20180109T0000Z'
fcast_time_obj =  datetime.datetime.strptime(fcast_time, '%Y%m%dT%H%MZ')

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

fnames_list = ['{im}_{datetime}.jpg'.format(im = im_type, datetime = (fcast_time_obj + datetime.timedelta(hours = vt)).strftime('%Y%m%d%H%M'))
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
                forest.util.download_from_s3(s3_url, local_path)
            except:
                print("    Warning: file not downloaded successfully:", s3_url)

fname_key = 'local_path'


##   Load
# Load data using iris. 

# Write Simulated Imagery data to  a dictionary indexed by time string

datasets['simim']['data'] = {'W': {}, 'I': {}, 'V':{}}

for file_name in datasets['simim']['local_paths_list']:
    
    cube_time_td = datetime.timedelta(hours = int(file_name[-5:-3]))
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
                im_array_dict.update({file_name[-16:-4]: matplotlib.pyplot.imread(file_name)})
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
    matplotlib.pyplot.register_cmap(name='SPSgreyscale', cmap=SPSgreyscale)

loadct_SPSgreyscale()

plot_options = {'V': {'norm': matplotlib.colors.Normalize(0, 1),
                      'cmap': 'binary_r'},
                'W': {'norm': matplotlib.colors.Normalize(198, 308),
                      'cmap': 'SPSgreyscale'},
                'I': {'norm': matplotlib.colors.Normalize(198, 308),
                      'cmap': 'SPSgreyscale'}}


        

# Set the initial values to be plotted
init_time = '201801091200'
init_var = 'I'


## Display plots

# Create a plot object for the left model display
plot_obj_left = simim_sat_plot.SimimSatPlot(datasets,
                                            plot_options,
                                            'plot_sea_left' + bokeh_id,
                                            init_var,
                                            init_time,
                                            'simim'
                                            )

plot_obj_left.current_time = init_time
bokeh_img_left = plot_obj_left.create_plot()

# Create a plot object for the right model display
plot_obj_right = simim_sat_plot.SimimSatPlot(datasets,
                                             plot_options,
                                             'plot_sea_right' + bokeh_id,
                                             init_var,
                                             init_time,
                                             'himawari-8'
                                             )

plot_obj_right.current_time = init_time
bokeh_img_right = plot_obj_right.create_plot()

plot_obj_right.link_axes_to_other_plot(plot_obj_left)

plot_list1 = [plot_obj_left, plot_obj_right]
bokeh_imgs1 = [bokeh_img_left, bokeh_img_right]
control1 = simim_sat_control.SimimSatControl(datasets, init_time, fcast_time_obj, plot_list1, bokeh_imgs1)


try:
    bokeh_mode = os.environ['BOKEH_MODE']
except:
    bokeh_mode = 'server'    
    
if bokeh_mode == 'server':
    bokeh.plotting.curdoc().add_root(control1.main_layout)
elif bokeh_mode == 'cli':
    bokeh.io.show(control1.main_layout)

bokeh.plotting.curdoc().title = 'Model simulated imagery vs Himawari-8'    
