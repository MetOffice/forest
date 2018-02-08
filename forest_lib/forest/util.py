'''

'''

import time
import os
import urllib.request
import datetime
import dateutil.parser

import matplotlib.colors
import matplotlib.cm
import numpy

import iris

SEA_REGION_DICT = {'indonesia': [-15.1, 1.0865, 99.875, 120.111],
                   'malaysia': [-2.75, 10.7365, 95.25, 108.737],
                   'phillipines': [3.1375, 21.349, 115.8, 131.987],
                   'se_asia': [-18.0, 29.96, 90.0, 153.96],
                   }


def download_from_s3(s3_url, local_path):

    '''Download files from AWS S3 if not present

    '''

    if not os.path.isfile(local_path):
        print('retrieving file from {0}'.format(s3_url))
        urllib.request.urlretrieve(s3_url, local_path)
        print('file {0} downloaded'.format(local_path))

    else:
        print(local_path, ' - File already downloaded')


def get_radar_colours():

    '''set up radar colours

    '''

    radar_colours1 = [(144 / 255.0, 144 / 255.0, 144 / 255.0, 0.0),
                      (122 / 255.0, 147 / 255.0, 212 / 255.0, 0.9),
                      (82 / 255.0, 147 / 255.0, 212 / 255.0, 0.95),
                      (39 / 255.0, 106 / 255.0, 188 / 255.0, 1.0),
                      (31 / 555.0, 201 / 255.0, 26 / 255.0, 1.0),
                      (253 / 255.0, 237 / 255.0, 57 / 255.0, 1.0),
                      (245 / 255.0, 152 / 255.0, 0 / 255.0, 1.0),
                      (235 / 255.0, 47 / 255.0, 26 / 255.0, 1.0),
                      (254 / 255.0, 92 / 255.0, 252 / 255.0, 1.0),
                      (255 / 255.0, 255 / 255.0, 255 / 255.0, 1.0)]

    radar_levels = numpy.array(
        [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    cmap_radar, norm_radar = matplotlib.colors.from_levels_and_colors(
        radar_levels, radar_colours1, extend='max')
    cmap_radar.set_name = 'radar'
    return {'cmap': cmap_radar, 'norm': norm_radar}


def get_wind_colours():

    '''Setup colormap for displaying wind speeds, based on Spectral_r colormap

    '''

    cm1 = matplotlib.cm.get_cmap('Spectral_r')
    wind_colours = cm1(
        numpy.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))

    # specify wind levels in miles per hour
    wind_levels = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0]

    cmap_wind_speed, norm_wind_speed = \
        matplotlib.colors.from_levels_and_colors(wind_levels,
                                                 wind_colours,
                                                 extend='max')
    cmap_wind_speed.set_name = 'wind_speed'
    return {'cmap': cmap_wind_speed, 'norm': norm_wind_speed}


def get_cloud_colours():

    '''Setup colormap for displaying cloud fraction, based on Greys_r colormap

    '''

    cm1 = matplotlib.cm.get_cmap('Greys_r')
    cloud_colours = cm1(numpy.arange(0.0, 1.01, 0.11))
#    cloud_colours[0,3] = 0.0
#    cloud_colours[1,3] = 0.1

    # specify wind levels in miles per hour
    cloud_levels = numpy.arange(0.0, 1.01, 0.1)

    cmap_cloud_fraction, norm_cloud_fraction = \
        matplotlib.colors.from_levels_and_colors(cloud_levels,
                                                 cloud_colours)
    cmap_cloud_fraction.set_name = 'cloud_fraction'
    return {'cmap': cmap_cloud_fraction, 'norm': norm_cloud_fraction}


def get_air_pressure_colours():

    '''Setup colormap for displaying air pressure , based on BuGn colormap

    '''

    cm1 = matplotlib.cm.get_cmap('BuGn')
    ap_colors = cm1(numpy.array(numpy.arange(0.0, 1.01, 1.0 / 12.0)))

    # specify wind levels in miles per hour
    ap_levels = numpy.arange(970.0, 1030.0, 5.0)

    cmap_ap, norm_ap = matplotlib.colors.from_levels_and_colors(ap_levels,
                                                                ap_colors,
                                                                extend='both')
    cmap_ap.set_name = 'air_pressure'
    return {'cmap': cmap_ap, 'norm': norm_ap}


def get_air_temp_colours():

    '''Setup colormap for displaying air pressure , based on viridis colormap

    '''

    cm1 = matplotlib.cm.get_cmap('viridis')
    air_temp_colours = cm1(numpy.array(numpy.arange(0.0, 1.01, 1.0 / 10.0)))

    # specify wind levels in miles per hour
    air_temp_levels = numpy.array(
        [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0])

    cmap_air_temp, norm__air_temp = \
        matplotlib.colors.from_levels_and_colors(air_temp_levels,
                                                 air_temp_colours,
                                                 extend='both')
    cmap_air_temp.set_name = 'air_temp'
    return {'cmap': cmap_air_temp, 'norm': norm__air_temp}


def get_time_str(time_in_hrs):

    '''
    Create formatted human-readable date/time string, calculated from
    epoch time in hours.
    '''

    datestamp1_raw = time.gmtime(time_in_hrs * 3600)
    datestr1 = \
        '{0:04d}-{1:02d}-{2:02d} {3:02d}h{4:02d}Z'.format(
            datestamp1_raw.tm_year,
            datestamp1_raw.tm_mon,
            datestamp1_raw.tm_mday,
            datestamp1_raw.tm_hour,
            datestamp1_raw.tm_min)
    return datestr1


def convert_vector_to_mag_angle(U, V):

    '''Convert U, V to magnitude and angle

    '''

    mag = numpy.sqrt(U ** 2 + V ** 2)
    angle = (numpy.pi / 2.) - numpy.arctan2(U / mag, V / mag)
    return mag, angle


def calc_wind_vectors(wind_x, wind_y, sf):

    '''Given cubes of x-wind and y-wind, subsample grid based on a scale
    factor (sf) and calculate the magnitude and angle of the vectors at each
    point on the grid.
    '''

    wv_dict = {}

    longitude_pts = wind_x.coord('longitude').points
    latitude_pts = wind_x.coord('latitude').points
    X, Y = numpy.meshgrid(longitude_pts, latitude_pts)
    X = X[::sf, ::sf]
    Y = Y[::sf, ::sf]
    wv_dict['wv_X_grid'] = X
    wv_dict['wv_Y_grid'] = Y
    wv_dict['wv_X'] = X[0, :]
    wv_dict['wv_Y'] = Y[:, 0]
    wv_dict['wv_U'] = wind_x.data[:, ::sf, ::sf]
    wv_dict['wv_V'] = wind_y.data[:, ::sf, ::sf]
    wind_mag, wind_angle = convert_vector_to_mag_angle(wv_dict['wv_U'],
                                                       wv_dict['wv_V'])
    # where wind speed is zero, there is an error calculating angle,
    # so set angle to 0.0 as it has physical meaning where speed is zero.
    wind_angle[numpy.isnan(wind_angle)] = 0.0
    wv_dict['wv_mag'] = wind_mag
    wv_dict['wv_angle'] = wind_angle

    return wv_dict


def create_colour_opts(var_list):

    '''Create a dictionary of plot options for use with matplotlib library for
    each of the standard plot types.
    '''

    col_opts_dict = dict([(s1, None) for s1 in var_list])
    col_opts_dict['precipitation'] = get_radar_colours()
    col_opts_dict['air_temperature'] = get_air_temp_colours()
    col_opts_dict['wind_vectors'] = get_wind_colours()
    col_opts_dict['wind_streams'] = get_wind_colours()
    col_opts_dict['wind_mslp'] = get_wind_colours()
    col_opts_dict['mslp'] = get_air_pressure_colours()
    col_opts_dict['cloud_fraction'] = get_cloud_colours()
    return col_opts_dict


def loadct_SPSgreyscale(do_register):

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
    ncolours = (n - color_min) - (n - color_max)
    cdict = {'red': [(0.0, cdict_max, cdict_max),
                     (1.0, cdict_min, cdict_min)],
             'green': [(0.0, cdict_max, cdict_max),
                       (1.0, cdict_min, cdict_min)],
             'blue': [(0.0, cdict_max, cdict_max),
                      (1.0, cdict_min, cdict_min)]}
    SPSgreyscale = matplotlib.colors.LinearSegmentedColormap(
        'SPSgreyscale', cdict, int(ncolours))
    if do_register:
        matplotlib.pyplot.register_cmap(name='SPSgreyscale', cmap=SPSgreyscale)
    return SPSgreyscale


def get_sat_simim_colours():

    '''

    '''

    cmap_sat_simim = loadct_SPSgreyscale(False)
    norm_sat_simim = matplotlib.colors.Normalize(198, 308)

    return {'cmap': cmap_sat_simim,
            'norm': norm_sat_simim}


def create_satellite_simim_plot_opts():

    '''

    '''

    plot_options = {'V': {'norm': matplotlib.colors.Normalize(0, 1),
                          'cmap': 'binary_r'},
                    'W': get_sat_simim_colours(),
                    'I': get_sat_simim_colours()}

    return plot_options


def extract_region(selected_region, ds1):

    '''Function to extract a regional subset of an iris cube based on latitude
    and longitude constraints.
    '''

    def get_lat(cell):

        '''

        '''
        region_bounds = region_dict[selected_region]
        return region_bounds[0] < cell < region_bounds[1]

    def get_long(cell):

        '''

        '''
        region_bounds = region_dict[selected_region]
        return region_bounds[2] < cell < region_bounds[3]

    con1 = iris.Constraint(latitude=get_lat,
                           longitude=get_long,
                           )
    return ds1.extract(con1)


def get_image_array_from_figure(fig):

    """Get the RGB buffer from the matplotlib figure.

    """

    h, w = fig.canvas.get_width_height()
    print(' width={0}\n height={1}'.format(w, h))

    fig.canvas.draw()

    buf = numpy.fromstring(fig.canvas.tostring_argb(), dtype=numpy.uint8)
    print('buf shape', buf.shape)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to
    # have it in RGBA mode
    buf = numpy.roll(buf, 3, axis=2)
    buf = numpy.flip(buf, axis=0)
    return buf


def get_model_run_times(num_days):

    """
    Create a list of model times from the last num_days days
    """

    lastweek = datetime.datetime.now() + datetime.timedelta(days=-num_days)
    lw_mn_str = '{dt.year:04}{dt.month:02}{dt.day:02}T0000Z'.format(
        dt=lastweek)
    lw_midnight = dateutil.parser.parse(str(lw_mn_str))
    fmt_str = '{dt.year:04}{dt.month:02}{dt.day:02}' + \
              'T{dt.hour:02}{dt.minute:02}Z'

    forecast_datetimes = [
        lw_midnight + datetime.timedelta(hours=step1)
        for step1 in range(0, 144, 12)]
    forecast_dt_str_list = [
        fmt_str.format(dt=dt1) for dt1 in forecast_datetimes]
    return forecast_datetimes, forecast_dt_str_list


def check_remote_file_exists(remote_path):
    """
    CHeck whether a file at the remote location specified by remore path
    exists by trying to open a url request.
    """

    file_exists = False
    try:
        _ = urllib.request.urlopen(remote_path)
        print('file {0} found at remote location.'.format(remote_path))
        file_exists = True
    except urllib.error.HTTPError:
        warning_msg1 = 'warning: file {0} NOT found at remote location.'
        warning_msg1 = warning_msg1.format(remote_path)
        print(warning_msg1)

    return file_exists
