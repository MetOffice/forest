"""Module for containing miscellaneous Forest functions.

Functions
---------

- download_from_s3() --
- get_radar_colours() -- Set dictionary of precipitation colormap.
- get_wind_colours() -- Set dictionary of wind speed colormap.
- get_cloud_colours() -- Set dictionary of cloud fraction colormap.
- get_air_pressure_colours() -- Set dictionary of mslp colormap.
- get_air_temp_colours() -- Set dictionary of temperature colormap.
- get_time_str() -- Convert epoch time into string.
- convert_vector_to_mag_angle() -- Convert winds, Cartesian to Polar.
- calc_wind_vectors() -- Calculate wind vectors from components.
- create_colour_opts() -- Create dict of colormaps.
- get_sat_simim_colours() -- Set dictionary of Sim Im colormap.
- create_satellite_simim_plot_opts() -- Create dict of colormaps.
- extract_region() -- Subset a cube into a restricted region.
- get_image_array_from_figure() -- Convert figure into RGBA array.
- check_remote_file_exists() -- Check if remote file exists.
- timer() -- Timer function for testing other functions.

"""
import time
import os
import datetime
import matplotlib.colors
import matplotlib.cm
import numpy
import iris
import bokeh.models.widgets
import bokeh.layouts


SEA_REGION_DICT = {'indonesia': [-15.1, 1.0865, 99.875, 120.111],
                   'malaysia': [-2.75, 10.7365, 95.25, 108.737],
                   'phillipines': [3.1375, 21.349, 115.8, 131.987],
                   'se_asia': [-18.0, 29.96, 90.0, 153.96],
                   }


def get_radar_colours():
    """Return dictionary of precip. colormap and normalisation."""
    radar_levels = numpy.array(
        [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    cmap_radar, norm_radar = matplotlib.colors.from_levels_and_colors(
        radar_levels, radar_colours(), extend='max')
    cmap_radar.set_name = 'radar'
    return {'cmap': cmap_radar, 'norm': norm_radar}


def radar_colors():
    return [
        (220 / 255.0, 220 / 255.0, 220 / 255.0, 1.0),
        (122 / 255.0, 147 / 255.0, 212 / 255.0, 0.9),
        (82 / 255.0, 147 / 255.0, 212 / 255.0, 0.95),
        (39 / 255.0, 106 / 255.0, 188 / 255.0, 1.0),
        (31 / 255.0, 201 / 255.0, 26 / 255.0, 1.0),
        (253 / 255.0, 237 / 255.0, 57 / 255.0, 1.0),
        (245 / 255.0, 152 / 255.0, 0 / 255.0, 1.0),
        (235 / 255.0, 47 / 255.0, 26 / 255.0, 1.0),
        (254 / 255.0, 92 / 255.0, 252 / 255.0, 1.0),
        (255 / 255.0, 255 / 255.0, 255 / 255.0, 1.0)
    ]


def get_wind_colours():

    """Return dictionary of wind speed colormap and normalisation."""


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

    """Return dictionary of cloud fraction colormap and normalisation."""

    cm1 = matplotlib.cm.get_cmap('Greys_r')
    cloud_colours = cm1(numpy.arange(0.0, 1.01, 0.11))

    # Specify cloud fraction levels
    cloud_levels = numpy.arange(0.0, 1.01, 0.1)

    cmap_cloud_fraction, norm_cloud_fraction = \
        matplotlib.colors.from_levels_and_colors(cloud_levels,
                                                 cloud_colours)
    cmap_cloud_fraction.set_name = 'cloud_fraction'
    
    return {'cmap': cmap_cloud_fraction, 'norm': norm_cloud_fraction}


def get_air_pressure_colours():

    """Return dictionary of MSLP colormap and normalisation."""

    cm1 = matplotlib.cm.get_cmap('BuGn')
    ap_colors = cm1(numpy.array(numpy.arange(0.0, 1.01, 1.0 / 12.0)))

    # Specify MSLP levels
    ap_levels = numpy.arange(970.0, 1030.0, 5.0)

    cmap_ap, norm_ap = matplotlib.colors.from_levels_and_colors(ap_levels,
                                                                ap_colors,
                                                                extend='both')
    cmap_ap.set_name = 'air_pressure'
    
    return {'cmap': cmap_ap, 'norm': norm_ap}


def get_air_temp_colours():

    """Return dictionary of air temp. colormap and normalisation."""

    cm1 = matplotlib.cm.get_cmap('viridis')
    air_temp_colours = cm1(numpy.array(numpy.arange(0.0, 1.01, 0.1)))

    # Specify temperature levels
    air_temp_levels = numpy.array(
        [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0])

    cmap_air_temp, norm__air_temp = \
        matplotlib.colors.from_levels_and_colors(air_temp_levels,
                                                 air_temp_colours,
                                                 extend='both')
    cmap_air_temp.set_name = 'air_temp'
    return {'cmap': cmap_air_temp, 'norm': norm__air_temp}


def get_time_str(time_in_hrs):

    """Convert epoch time into string.
    
    Create formatted human-readable date/time string, calculated from
    epoch time in hours.
    
    Arguments
    ---------
    
    - time_in_hrs -- Epoch time variable.
    
    """

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

    """Convert U, V to magnitude and angle.
    
    Arguments
    ---------
    
    - U -- Zonal wind component array.
    - V -- Meridional wind component array.

    """
    grid_size = (U.shape[0],U.shape[1])

    mag=create_zeros_cube(grid_size, U)
    angle=create_zeros_cube(grid_size, U)

    mag.data = numpy.sqrt(U.data ** 2 + V.data ** 2)
    angle.data = (numpy.pi / 2.) - numpy.arctan2(U.data / mag.data, V.data / mag.data)
    
    return mag, angle


def calc_wind_vectors(wind_x, wind_y, wind_grid_size):
    """Return wind vector dict calculated from wind components.

    Given cubes of x-wind and y-wind, subsample grid based on a scale
    factor (sf) and calculate the magnitude and angle of the vectors at each
    point on the grid.

    Arguments
    ---------
    - wind_x -- Zonal wind component cube.
    - wind_y -- Meridional wind component cube.
    """
    wv_dict = {}
    wv_dict['wv_U'] = do_regrid(wind_grid_size, wind_x)
    wv_dict['wv_V'] = do_regrid(wind_grid_size, wind_y)

    longitude_pts = wv_dict['wv_U'].coord('longitude').points
    latitude_pts = wv_dict['wv_U'].coord('latitude').points
    X, Y = numpy.meshgrid(longitude_pts, latitude_pts)
    wv_dict['wv_X_grid'] = iris.cube.Cube(X)
    wv_dict['wv_Y_grid'] = iris.cube.Cube(Y)
    wv_dict['wv_X'] = wv_dict['wv_X_grid'][0, :]
    wv_dict['wv_Y'] = wv_dict['wv_Y_grid'][:, 0]

    wind_mag, wind_angle = convert_vector_to_mag_angle(wv_dict['wv_U'],
                                                       wv_dict['wv_V'])

    # Where wind speed is zero, there is an error calculating angle,
    # so set angle to 0.0 as it has physical meaning where speed is zero.
    wind_angle.data[numpy.isnan(wind_angle.data)] = 0.0
    wv_dict['wv_mag'] = wind_mag
    wv_dict['wv_angle'] = wind_angle
    return wv_dict


def create_zeros_cube(grid_size, cube1):
    num_lat = grid_size[0]
    num_long = grid_size[1]


    min_lat = cube1.coord('latitude').points[0]
    max_lat = cube1.coord('latitude').points[-1]
    min_long = cube1.coord('longitude').points[0]
    max_long = cube1.coord('longitude').points[-1]

    # create the arrays of latitude and longitude coordinates for the regridded cube
    step_lat = (max_lat - min_lat) / num_lat
    step_long = (max_long - min_long) / num_long
    lat_arr1 = numpy.array([min_lat + i1*step_lat for i1 in range(0,num_lat)])
    long_arr1 = numpy.array([min_long + i1*step_long for i1 in range(0,num_long)])


    current_lat_coord = cube1.coords('latitude')[0]
    current_long_coord = cube1.coords('longitude')[0]

    # create the iris coordinates
    lat_coord = \
        iris.coords.DimCoord(lat_arr1,
                             standard_name=current_lat_coord.standard_name,
                             coord_system=current_lat_coord.coord_system,
                             units=current_lat_coord.units)
    long_coord = \
        iris.coords.DimCoord(long_arr1,
                             standard_name=current_long_coord.standard_name,
                             coord_system=current_long_coord.coord_system,
                             units=current_long_coord.units)

    # create a dummy cube of the correct size and add the coordinates
    c1 = iris.cube.Cube(numpy.zeros((num_lat, num_long)))
    c1.add_dim_coord(lat_coord, 0)
    c1.add_dim_coord(long_coord, 1)
    c1.units=None
    return c1

def do_regrid(grid_size, cube1):
    # specify the dimensions of the regridded cube
    c1 = create_zeros_cube(grid_size, cube1)

    # specify operation parameters
    regrid_method = iris.analysis.Linear()

    # perform the regridding operation
    return cube1.regrid(c1, regrid_method)


def create_colour_opts(var_list):

    """Return dictionary of variable colormaps functions.
    
    Create a dictionary of plot options for use with matplotlib library for
    each of the standard plot types.
    
    Arguments
    ---------
    
    - var_list -- List of variables to create colormap dict keys for.
    
    """

    col_opts_dict = dict([(s1, None) for s1 in var_list])
    col_opts_dict['precipitation'] = get_radar_colours()
    col_opts_dict['accum_precip_3hr'] = get_radar_colours()
    col_opts_dict['accum_precip_6hr'] = get_radar_colours()
    col_opts_dict['accum_precip_12hr'] = get_radar_colours()
    col_opts_dict['accum_precip_24hr'] = get_radar_colours()
    col_opts_dict['air_temperature'] = get_air_temp_colours()
    col_opts_dict['wind_vectors'] = get_wind_colours()
    col_opts_dict['wind_streams'] = get_wind_colours()
    col_opts_dict['wind_mslp'] = get_wind_colours()
    col_opts_dict['mslp'] = get_air_pressure_colours()
    col_opts_dict['cloud_fraction'] = get_cloud_colours()
    
    return col_opts_dict


def get_sat_simim_colours(min_bt, max_bt):
    
    """Return dictionary of MSLP colormap and normalisation.
    
    In a 256 colour setup, remove 4 colours (black and the three
    darkest greys) from the black end of the greyscale and one (white)
    from the white end.
    
    Arguments
    ---------
    
    - min_bt -- Numeric; Lower limit of brightness temp. colormap.
    - max_bt -- Numeric; Upper limit of brightness temp. colormap.
    
    """
    
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

    cmap_sat_simim = SPSgreyscale
    norm_sat_simim = matplotlib.colors.Normalize(min_bt, max_bt)

    return {'cmap': cmap_sat_simim, 'norm': norm_sat_simim}


def create_satellite_simim_plot_opts():

    """Return dictionary of Sim Im colormap functions."""

    plot_options = {'V': {'norm': matplotlib.colors.Normalize(0, 1),
                          'cmap': 'binary_r'},
                    'W': get_sat_simim_colours(198, 273),
                    'I': get_sat_simim_colours(198, 308)}

    return plot_options


def extract_region(region_dict, selected_region, ds1):

    """Extract sub-region from cube data.
    
    Function to extract a regional subset of an iris cube based on latitude
    and longitude constraints.
    
    Arguments
    ---------
    
    - region_dict -- Dict; Dictionary of regions and their limits.
    - selected_region -- Str; Key to region_dict for selected region.
    - ds1 -- Cube; Data cube to constrain.
    
    """

    def get_lat(cell):

        """Set latitude constraint"""
        
        region_bounds = region_dict[selected_region]
        return region_bounds[0] < cell < region_bounds[1]

    def get_long(cell):

        """Set longitude constraint"""
        
        region_bounds = region_dict[selected_region]
        return region_bounds[2] < cell < region_bounds[3]

    con1 = iris.Constraint(latitude=get_lat, longitude=get_long)
    
    return ds1.extract(con1)


def get_image_array_from_figure(fig):

    """Get the RGB buffer from the matplotlib figure.
    
    Arguments
    ---------
    
    - fig -- matplotlib Figure; Figure to convert into buffer array.
    
    """

    h, w = fig.canvas.get_width_height()

    fig.canvas.draw()

    buf = numpy.fromstring(fig.canvas.tostring_argb(), dtype=numpy.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb gives pixmap in ARGB format. 
    # Roll the ALPHA channel to convert to RGBA format.
    buf = numpy.roll(buf, 3, axis=2)
    buf = numpy.flipud(buf)
    
    return buf


def timer(func):
    """Timer function.

        Arguments
    ---------
    - func -- Function; Function to test.
    """
    def timed_func(*args, **kwargs):
        """Times other functions."""
        start_time = time.time()
        ret_val = func(*args, **kwargs)
        end_time = time.time()
        duration_in_seconds = end_time - start_time
        print('function {0} ran for a duration of {1}.seconds'.format(str(func),
                                                                      duration_in_seconds))
        return ret_val
    return timed_func


def counter(func):
    """Decorator to report of function invocation"""
    count = 0
    def wrapper(*args, **kwargs):
        nonlocal count
        count += 1
        print(str(func), "called :", count, "times")
        return func(*args, **kwargs)
    return wrapper


def load_error_page():
    msg1 = 'Error: data not found'
    error_msg_widget = bokeh.models.widgets.Div(text=msg1,
                                                 height=200,
                                                 width=400,
                                                 )
    layout1 = bokeh.layouts.column(error_msg_widget)
    return layout1
