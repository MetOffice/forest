import sys

import time
import os

if sys.version[0] == '3':
    import urllib.request

import matplotlib.colors
import matplotlib.cm
import numpy

import iris


def download_from_s3(s3_url, local_path):
    '''
    Download files from AWS S3 if not present
    '''
    if not os.path.isfile(local_path):
        print('retrieving file from {0}'.format(s3_url))
        urllib.request.urlretrieve(s3_url, local_path) 
        print('file {0} downloaded'.format(local_path))

    else:
        print(local_path, ' - File already downloaded')


def get_radar_colours():
    '''
    set up radar colours
    '''
    radar_colours1 = [(144/255.0, 144/255.0, 144/255.0,0.0),
                     (122/255.0,147/255.0,212/255.0,0.9),
                     (82/255.0,147/255.0,212/255.0,0.95),
                     (39/255.0,106/255.0,188/255.0,1.0),
                     (31/555.0,201/255.0,26/255.0,1.0),
                     (253/255.0,237/255.0,57/255.0,1.0),
                     (245/255.0,152/255.0,0/255.0,1.0),
                     (235/255.0,47/255.0,26/255.0,1.0),
                     (254/255.0,92/255.0,252/255.0,1.0),
                     (255/255.0,255/255.0,255/255.0,1.0)]


    radar_levels = numpy.array([0.0,0.1,0.25,0.5,1.0,2.0,4.0,8.0,16.0,32.0]) 
    cmap_radar, norm_radar = matplotlib.colors.from_levels_and_colors(radar_levels, radar_colours1, extend='max')
    cmap_radar.set_name = 'radar'
    
    return {'cmap':cmap_radar,'norm':norm_radar}

   
def get_time_str(time_in_hrs):
    '''
    Create formatted human-readable date/time string, calculated from epoch time in hours.
    '''
    datestamp1_raw = time.gmtime(time_in_hrs*3600)
    datestr1 = '{0:04d}-{1:02d}-{2:02d} {3:02d}h{4:02d}Z'.format(datestamp1_raw.tm_year,
                                        datestamp1_raw.tm_mon,
                                        datestamp1_raw.tm_mday,
                                        datestamp1_raw.tm_hour,
                                        datestamp1_raw.tm_min) 
    return datestr1


def create_plot_opts_dict(var_list):
    '''
    Create a dictionary of plot options for use with holoviews library for each of the
    standard plot types.
    '''
    plot_opts_dict = dict([(s1,None) for s1 in var_list])

    plot_opts_dict['precipitation'] = {'Image': {'plot':{'colorbar':True,
                                                         'width':1000,
                                                         'height':800},
                                                 'style':get_radar_colours(),
                                                }
                                      }
    return plot_opts_dict


def create_colour_opts(var_list):
    '''
    Create a dictionary of plot options for use with matplotlib library for each of the
    standard plot types.
    '''
    col_opts_dict = dict([(s1,None) for s1 in var_list])
    col_opts_dict['precipitation'] = get_radar_colours()
    
    return col_opts_dict
    
    
def get_image_array_from_figure(fig):

    '''
    
    '''
    
    width, height = fig.get_size_inches() * fig.get_dpi()
    
    # Get the RGB buffer from the figure
    h, w = fig.canvas.get_width_height()
    print(' width={0}\nheight={1}'.format(w,h))
    buf = numpy.fromstring(fig.canvas.tostring_argb(), dtype=numpy.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to 
    #  have it in RGBA mode
    buf = numpy.roll(buf, 3, axis = 2 )
    buf = buf[::-1, :, :]
    
    return buf