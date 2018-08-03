import textwrap
import dateutil

import numpy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot
import matplotlib.cm

import cartopy
import cartopy.crs

import bokeh.models.widgets
import bokeh.plotting

import forest.util
import forest.data

import numpy as np
import scipy.ndimage
import iris.analysis


def rgba_from_mappable(mappable, shape2d):
    """Convert matplotlib scalar mappable to RGBA

    .. note:: 2D shape needs to be provided to reconstruct
              image array

    :param mappable: matplotlib ScalarMappable instance
    :param shape2d: tuple describing 2D shape of image
    :returns: array np.uint8 suitable for use with image_rgba()
    """
    ni, nj = shape2d
    return mappable.to_rgba(mappable.get_array(),
                            bytes=True).reshape((ni, nj, 4))


def coastlines(figure, scale="110m", extent=None):
    """Add cartopy coastline to a figure

    Translates cartopy.feature.COASTLINE object
    into collection of bokeh lines

    .. note:: This method assumes the map projection
              is cartopy.crs.PlateCarreee

    :param figure: bokeh figure instance
    :param scale: cartopy coastline scale '110m', '50m' or '10m'
    :param extent: x_start, x_end, y_start, y_end
    """
    coastline = cartopy.feature.COASTLINE
    coastline.scale = scale
    for geometry in coastline.geometries():
        x, y = geometry[0].xy
        if extent is not None:
            x, y = clip_xy(x, y, extent)
        if x.shape[0] == 0:
            continue
        figure.line(x, y,
                    color='black',
                    level='overlay')


def clip_xy(x, y, extent):
    """Clip coastline to be inside figure extent"""
    x, y = np.asarray(x), np.asarray(y)
    x_start, x_end, y_start, y_end = extent
    pts = np.where((x > x_start) &
                   (x < x_end) &
                   (y > y_start) &
                   (y < y_end))
    return x[pts], y[pts]


BOKEH_TOOLS_LIST = ['pan','wheel_zoom','reset','save','box_zoom','hover']

class MissingDataError(Exception):
    def __init__(self, config, var, time):
        self.config = config
        self.var = var
        self.time = time


def pretty_bokeh_figure(*args, **kwargs):
    """Helper to make prettier bokeh figures"""
    figure = bokeh.plotting.figure(*args, **kwargs)

    # Extra x-axis above figure
    position = 'above'
    figure.extra_x_ranges[position] = figure.x_range
    axis = bokeh.models.LinearAxis(x_range_name=position)
    axis.major_label_text_font_size = '0pt'
    figure.add_layout(axis, position)

    # Extra y-axis right of figure
    position = 'right'
    figure.extra_y_ranges[position] = figure.y_range
    axis = bokeh.models.LinearAxis(y_range_name=position)
    axis.major_label_text_font_size = '0pt'
    figure.add_layout(axis, position)
    return figure


def smooth_image(array, output_shape):
    """Smooth high resolution imagery"""
    max_ni, max_nj = output_shape
    ni, nj, _ = array.shape
    # scipy docs: int(round(factor * n)) = max_n
    factor = max_ni / ni, max_nj / nj
    output_array = np.zeros((max_ni, max_nj, 4), dtype=np.uint8)
    for i in range(4):
        output_array[:, :, i] = scipy.ndimage.zoom(array[:, :, i], factor)
    return output_array


class ForestPlot(object):
    '''
    Main plot class. The plotting function is create_plot().
    '''
    TITLE_TEXT_WIDTH = 40
    PRESSURE_LEVELS_HPA = range(980, 1030, 2)

    def __init__(self,
                 dataset,
                 model_run_time,
                 plot_options,
                 figure_name,
                 plot_var,
                 conf1,
                 reg1,
                 rd1,
                 unit_dict,
                 unit_dict_display,
                 app_path,
                 init_time,
                 bokeh_figure=None,
                 visible=True):
        '''Initialisation function for ForestPlot class
        '''
        projection = cartopy.crs.PlateCarree()
        self.current_figure = matplotlib.pyplot.figure(figure_name)
        self.current_axes = self.current_figure.add_subplot(111, projection=projection)

        self.region_dict = rd1
        self.main_plot = None
        self.current_time = init_time
        self.plot_options = plot_options
        self.dataset = dataset
        self.model_run_time = model_run_time
        self.current_var = plot_var
        self._set_config_value(conf1)
        self.current_region = reg1
        self.app_path = app_path
        self.data_bounds = self.region_dict[self.current_region]
        self.selected_point = None
        self.setup_plot_funcs()
        self.setup_pressure_labels()
        self.current_title = ''
        self.stats_string = ''
        self.colorbar_link = plot_var + '_colorbar.png'
        if bokeh_figure is None:
            cur_region = self.region_dict[self.current_region]
            assert len(cur_region) == 4, "must provide y_start, y_end, x_start, x_end"
            y_start = cur_region[0]
            y_end = cur_region[1]
            x_start = cur_region[2]
            x_end = cur_region[3]
            x_range = bokeh.models.Range1d(x_start, x_end, bounds=(x_start, x_end))
            y_range = bokeh.models.Range1d(y_start, y_end, bounds=(y_start, y_end))
            bokeh_figure = bokeh.plotting.figure(plot_width=800,
                                                 plot_height=600,
                                                 x_range=x_range,
                                                 y_range=y_range,
                                                 tools=','.join(BOKEH_TOOLS_LIST),
                                                 toolbar_location='above')
        self.bokeh_figure = bokeh_figure
        self.bokeh_image = None
        self.bokeh_img_ds = None
        self.unit_dict = unit_dict
        self.unit_dict_display = unit_dict_display
        self.stats_widget = None
        self.colorbar_widget = None

        self.coast_res = '110m'

        self.visible = visible
        self._shape2d = None

    @property
    def current_img_array(self):
        """current image array taken from matplotlib figure canvas"""
        rgba_source = "mappable"
        if self._shape2d is None:
            return None
        if rgba_source == "canvas":
            return forest.util.get_image_array_from_figure(self.current_figure)
        else:
            # HACK: self._shape2d is populated by self.get_data()
            ni, nj = self._shape2d
            array = rgba_from_mappable(self.main_plot, (ni - 1, nj - 1))

            # Smooth high resolution imagery
            max_ni, max_nj = 800, 600
            ni, nj, _ = array.shape
            if (ni > max_ni) or (nj > max_nj):
                return smooth_image(array, (max_ni, max_nj))
            else:
                return array

    def _set_config_value(self, new_config):
        self.current_config = new_config
        self.plot_description = self.dataset[
            self.current_config]['data_type_name']

    def setup_pressure_labels(self):
        '''Create dict of pressure levels, to be used labelling MSLP contour
        plots.
        '''
        self.mslp_contour_label_dict = {}
        for pressure1 in ForestPlot.PRESSURE_LEVELS_HPA:
            self.mslp_contour_label_dict[
                pressure1] = '{0:d}hPa'.format(int(pressure1))

    def setup_plot_funcs(self):
        '''Set up dictionary of plot functions. This is used by the main
        create_plot() function to call the plotting function relevant to the
        specific variable being plotted. There is also a second dictionary
        which is by the update_plot() function, which does the minimum amount
        of work to update the plot, and is used for some option changes,
        mainly a change in the forecast time selected.
        '''
        self.plot_funcs = {'precipitation': self.plot_precip,
                           'accum_precip_3hr': self.plot_precip,
                           'accum_precip_6hr': self.plot_precip,
                           'accum_precip_12hr': self.plot_precip,
                           'accum_precip_24hr': self.plot_precip,
                           'wind_vectors': self.plot_wind_vectors,
                           'wind_mslp': self.plot_wind_mslp,
                           'wind_streams': self.plot_wind_streams,
                           'mslp': self.plot_mslp,
                           'air_temperature': self.plot_air_temp,
                           'cloud_fraction': self.plot_cloud,
                           'himawari-8': self.plot_him8,
                           'simim': self.plot_simim,
                           'W': self.plot_sat_simim_imagery,
                           'I': self.plot_sat_simim_imagery,
                           'V': self.plot_sat_simim_imagery,
                           'blank': self.create_blank,
                           }

        self.update_funcs = {'precipitation': self.update_precip,
                             'accum_precip_3hr': self.update_precip,
                             'accum_precip_6hr': self.update_precip,
                             'accum_precip_12hr': self.update_precip,
                             'accum_precip_24hr': self.update_precip,
                             'wind_vectors': self.update_wind_vectors,
                             'wind_mslp': self.update_wind_mslp,
                             'wind_streams': self.update_wind_streams,
                             'mslp': self.update_mslp,
                             'air_temperature': self.update_air_temp,
                             'cloud_fraction': self.update_cloud,
                             'himawari-8': self.update_him8,
                             'simim': self.update_simim,
                             'W': self.update_sat_simim_imagery,
                             'I': self.update_sat_simim_imagery,
                             'V': self.update_sat_simim_imagery,
                             'blank': self.create_blank,
                             }
        self.stats_data_var = dict([(k1,k1) for k1 in self.plot_funcs.keys()])
        self.stats_data_var['wind_vectors'] = forest.data.WIND_SPEED_NAME
        self.stats_data_var['wind_mslp'] = forest.data.WIND_SPEED_NAME
        self.stats_data_var['wind_streams'] = forest.data.WIND_SPEED_NAME

    def update_coords(self, data_cube):
        '''Update the latitude and longitude coordinates for the data.
        '''
        self.coords_lat = data_cube.coords('latitude')[0].points
        self.coords_long = data_cube.coords('longitude')[0].points

    def create_blank(self):
        self.main_plot = None
        self.current_title = 'Blank plot'

    def get_data(self, var_name=None):
        config_data = self.dataset[self.current_config]['data']
        if var_name:
            data_cube = config_data.get_data(var_name=var_name,
                                 selected_time = self.current_time)
        else:
            data_cube = config_data.get_data(var_name=self.current_var,
                                             selected_time=self.current_time)

        # HACK: cache image array shape after get_data()
        self._shape2d = data_cube.data.shape
        return data_cube

    def update_precip(self):
        '''Update function for precipitation plots, called by update_plot() when
        precipitation is the selected plot type.
        '''
        data_cube = self.get_data()
        array_for_update = data_cube.data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(data_cube)
        self.update_stats(data_cube)

    def plot_precip(self):
        '''Function for creating precipitation plots, called by create_plot when
        precipitation is the selected plot type.
        '''
        data_cube = self.get_data()
        cmap = self.plot_options[self.current_var]['cmap']
        norm = self.plot_options[self.current_var]['norm']
        self.update_coords(data_cube)
        self.main_plot = \
            self.current_axes.pcolormesh(self.coords_long,
                                         self.coords_lat,
                                         data_cube.data,
                                         cmap=cmap,
                                         norm=norm,
                                         edgecolor='face',
                                         transform=cartopy.crs.PlateCarree())
        self.update_title(data_cube)
        self.update_stats(data_cube)

    def update_wind_vectors(self):
        '''Update function for wind vector plots, called by update_plot() when
        wind vectors is the selected plot type.
        '''
        wind_speed_cube = self.get_data(forest.data.WIND_SPEED_NAME)

        array_for_update = wind_speed_cube.data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(wind_speed_cube)
        self.update_stats(wind_speed_cube)
        wv_u_data = self.get_data(var_name='wv_U').data
        wv_v_data = self.get_data(var_name='wv_V').data
        self.quiver_plot.set_UVC(wv_u_data,
                                 wv_v_data)

    def plot_wind_vectors(self):
        '''Function for creating wind vector plots, called by create_plot when
        wind vectors is the selected plot type.
        '''
        wind_speed_cube = self.get_data(forest.data.WIND_SPEED_NAME)
        self.update_coords(wind_speed_cube)
        self.main_plot = \
            self.current_axes.pcolormesh(self.coords_long,
                                         self.coords_lat,
                                         wind_speed_cube.data,
                                         cmap=self.plot_options[
                                             self.current_var]['cmap'],
                                         norm=self.plot_options[
                                             self.current_var]['norm']
                                         )
        self.quiver_plot = \
            self.current_axes.quiver(
                self.get_data('wv_X').data,
                self.get_data('wv_Y').data,
                self.get_data('wv_U').data,
                self.get_data('wv_V').data,
                units='height')
        qk = self.current_axes.quiverkey(self.quiver_plot,
                                         0.9,
                                         0.9,
                                         2,
                                         r'$2 \frac{m}{s}$',
                                         labelpos='E',
                                         coordinates='figure')
        self.update_title(wind_speed_cube)
        self.update_stats(wind_speed_cube)

    def update_wind_mslp(self):
        '''Update function for wind speed with MSLP contours plots, called by
        update_plot() when wind speed with MSLP is the selected plot type.
        '''
        wind_speed_cube = self.get_data(var_name=forest.data.WIND_SPEED_NAME)
        array_for_update = wind_speed_cube.data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        # to update contours, remove old elements and generate new contours
        for c1 in self.mslp_contour.collections:
            self.current_axes.collections.remove(c1)

        ap_cube = self.get_data(var_name=forest.data.MSLP_NAME)
        self.mslp_contour = \
            self.current_axes.contour(self.long_grid_mslp,
                                      self.lat_grid_mslp,
                                      ap_cube.data,
                                      levels=ForestPlot.PRESSURE_LEVELS_HPA,
                                      colors='k')
        self.current_axes.clabel(self.mslp_contour,
                                 inline=False,
                                 fmt=self.mslp_contour_label_dict)

        self.update_title(wind_speed_cube)
        self.update_stats(wind_speed_cube)

    def plot_wind_mslp(self):
        '''Function for creating wind speed with MSLP contour plots, called by
        create_plot when wind speed with MSLP contours is the selected plot
        type.
        '''
        wind_speed_cube = self.get_data(var_name=forest.data.WIND_SPEED_NAME)
        self.update_coords(wind_speed_cube)
        self.main_plot = \
            self.current_axes.pcolormesh(self.coords_long,
                                         self.coords_lat,
                                         wind_speed_cube.data,
                                         cmap=self.plot_options[
                                             self.current_var]['cmap'],
                                         norm=self.plot_options[
                                             self.current_var]['norm']
                                         )

        ap_cube = self.get_data(forest.data.MSLP_NAME)
        lat_mslp = ap_cube.coords('latitude')[0].points
        long_mslp = ap_cube.coords('longitude')[0].points
        self.long_grid_mslp, self.lat_grid_mslp = numpy.meshgrid(
            long_mslp, lat_mslp)
        self.mslp_contour = \
            self.current_axes.contour(self.long_grid_mslp,
                                      self.lat_grid_mslp,
                                      ap_cube.data,
                                      levels=ForestPlot.PRESSURE_LEVELS_HPA,
                                      colors='k')
        self.current_axes.clabel(self.mslp_contour,
                                 inline=False,
                                 fmt=self.mslp_contour_label_dict)
        self.update_title(wind_speed_cube)

    def update_wind_streams(self):
        '''Update function for wind streamline plots, called by update_plot()
        when wind streamlines is the selected plot type.
        '''
        wind_speed_cube = self.get_data(var_name=forest.data.WIND_SPEED_NAME)
        array_for_update = wind_speed_cube.data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(wind_speed_cube)
        self.update_stats(wind_speed_cube)

        # remove old plot elements if they are still present
        self.current_axes.collections.remove(self.wind_stream_plot.lines)
        for p1 in self.wind_stream_patches:
            self.current_axes.patches.remove(p1)

        pl1 = list(self.current_axes.patches)
        self.wind_stream_plot = \
            self.current_axes.streamplot(
                self.get_data(var_name='wv_X_grid').data,
                self.get_data(var_name='wv_Y_grid').data,
                self.get_data(var_name='wv_U').data,
                self.get_data(var_name='wv_V').data,
                color='k',
                density=[0.5, 1.0])
        # we need to manually keep track of arrows so they can be removed when
        # the plot is updated
        pl2 = list(self.current_axes.patches)
        self.wind_stream_patches = [p1 for p1 in pl2 if p1 not in pl1]

        self.update_stats(wind_speed_cube)

    def plot_wind_streams(self):
        '''Function for creating wind streamline plots, called by create_plot when
        wind streamlines is the selected plot type.
        '''
        wind_speed_cube = self.get_data(var_name=forest.data.WIND_SPEED_NAME)
        self.update_coords(wind_speed_cube)
        self.main_plot = \
            self.current_axes.pcolormesh(self.coords_long,
                                         self.coords_lat,
                                         wind_speed_cube.data,
                                         cmap=self.plot_options[
                                             self.current_var]['cmap'],
                                         norm=self.plot_options[
                                             self.current_var]['norm']
                                         )
        pl1 = list(self.current_axes.patches)

        self.wind_stream_plot = \
            self.current_axes.streamplot(
                self.get_data(var_name='wv_X_grid').data,
                self.get_data(var_name='wv_Y_grid').data,
                self.get_data(var_name='wv_U').data,
                self.get_data(var_name='wv_V').data,
                color='k',
                density=[0.5, 1.0])

        # we need to manually keep track of arrows so they can be removed when
        # the plot is updated
        pl2 = list(self.current_axes.patches)
        self.wind_stream_patches = [p1 for p1 in pl2 if p1 not in pl1]
        self.update_title(wind_speed_cube)
        self.update_stats(wind_speed_cube)


    def update_air_temp(self):
        '''Update function for air temperature plots, called by update_plot() when
        air temperature is the selected plot type.
        '''
        at_cube = self.get_data()
        array_for_update = at_cube.data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(at_cube)
        self.update_stats(at_cube)

    def plot_air_temp(self):
        '''Function for creating air temperature plots, called by create_plot when
        air temperature is the selected plot type.
        '''
        at_cube = self.get_data()
        self.update_coords(at_cube)
        self.main_plot = \
            self.current_axes.pcolormesh(self.coords_long,
                                         self.coords_lat,
                                         at_cube.data,
                                         cmap=self.plot_options[
                                             self.current_var]['cmap'],
                                         norm=self.plot_options[
                                             self.current_var]['norm']
                                         )
        self.update_title(at_cube)
        self.update_stats(at_cube)

    def update_mslp(self):
        '''Update function for MSLP plots, called by update_plot() when
        MSLP is the selected plot type.
        '''
        ap_cube = self.get_data()
        array_for_update = ap_cube.data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(ap_cube)
        self.update_stats(ap_cube)

    def plot_mslp(self):
        '''Function for creating MSLP plots, called by create_plot when
        MSLP is the selected plot type.
        '''
        ap_cube = self.get_data()
        self.update_coords(ap_cube)
        self.main_plot = \
            self.current_axes.pcolormesh(self.coords_long,
                                         self.coords_lat,
                                         ap_cube.data,
                                         cmap=self.plot_options[
                                             self.current_var]['cmap'],
                                         norm=self.plot_options[
                                             self.current_var]['norm']
                                         )
        self.update_title(ap_cube)
        self.update_stats(ap_cube)

    def update_cloud(self):
        '''Update function for cloud fraction plots, called by update_plot() when
        cloud fraction is the selected plot type.
        '''
        cloud_cube = self.get_data()
        array_for_update = cloud_cube.data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(cloud_cube)
        self.update_stats(cloud_cube)

    def plot_cloud(self):
        '''Function for creating cloud fraction plots, called by create_plot when
        cloud fraction is the selected plot type.
        '''
        cloud_cube = self.get_data()
        self.update_coords(cloud_cube)
        self.main_plot = \
            self.current_axes.pcolormesh(self.coords_long,
                                         self.coords_lat,
                                         cloud_cube.data,
                                         cmap=self.plot_options[
                                             self.current_var]['cmap'],
                                         norm=self.plot_options[
                                             self.current_var]['norm']
                                         )
        self.update_title(cloud_cube)
        self.update_stats(cloud_cube)

    def update_him8(self):
        '''Update function for himawari-8 image plots, called by update_plot()
        when cloud fraction is the selected plot type.
        '''
        him8_image = self.dataset[
            'himawari-8']['data'].get_data(self.current_var, selected_time=self.current_time)
        self.current_axes.images.remove(self.main_plot)
        self.main_plot = self.current_axes.imshow(him8_image,
                                                  extent=(self.data_bounds[2],
                                                          self.data_bounds[3],
                                                          self.data_bounds[0],
                                                          self.data_bounds[1]),
                                                  origin='upper')
        self.update_title(None)

    def plot_him8(self):
        '''Function for creating himawari-8 image plots, called by create_plot()
        when cloud fraction is the selected plot type.
        '''
        him8_data = self.dataset['himawari-8']['data']
        him8_image = him8_data.get_data(self.current_var,
                                        selected_time=self.current_time)
        self.main_plot = self.current_axes.imshow(him8_image,
                                                  extent=(self.data_bounds[2],
                                                          self.data_bounds[3],
                                                          self.data_bounds[0],
                                                          self.data_bounds[1]),
                                                  origin='upper')
        self.current_axes.set_extent((self.data_bounds[2],
                                      self.data_bounds[3],
                                      self.data_bounds[0],
                                      self.data_bounds[1]))

        self.update_title(None)

    def update_simim(self):
        '''Update function for himawari-8 image plots, called by update_plot()
        when cloud fraction is the selected plot type.
        '''
        simim_cube = self.dataset['simim']['data'].get_data(
            self.current_var, selected_time=self.current_time)
        array_for_update = simim_cube.data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(None)

    def plot_simim(self):
        '''Function for creating himawari-8 image plots, called by create_plot()
        when cloud fraction is the selected plot type.
        '''
        simim_cube = self.dataset['simim']['data'].get_data(self.current_var,
                                                            selected_time=self.current_time)
        lats = simim_cube.coord('grid_latitude').points
        lons = simim_cube.coord('grid_longitude').points
        self.main_plot = \
            self.current_axes.pcolormesh(lons,
                                         lats,
                                         simim_cube.data,
                                         cmap=self.plot_options[
                                             self.current_var]['cmap'],
                                         norm=self.plot_options[
                                             self.current_var]['norm']
                                         )
        self.current_axes.set_extent((self.data_bounds[2],
                                      self.data_bounds[3],
                                      self.data_bounds[0],
                                      self.data_bounds[1]))

        self.update_title(None)

    def update_sat_simim_imagery(self):
        if self.current_config == 'himawari-8':
            self.update_him8()
        elif self.current_config == 'simim':
            self.update_simim()

    def plot_sat_simim_imagery(self):
        if self.current_config == 'himawari-8':
            self.plot_him8()
        elif self.current_config == 'simim':
            self.plot_simim()

    def update_stats(self, current_cube):
        print("dummy update_stats called")

    def _update_stats(self, current_cube):
        data_to_process = current_cube.data
        stats_str_list = [self.current_title]
        unit_str = self.unit_dict_display[self.current_var]
        max_val = numpy.max(data_to_process)
        min_val = numpy.min(data_to_process)
        mean_val = numpy.mean(data_to_process)
        std_val = numpy.std(data_to_process)
        rms_val = numpy.sqrt(numpy.mean(numpy.power(data_to_process, 2.0)))
        model_run_info = 'Current model run start time: '
        mr_dtobj = dateutil.parser.parse(self.model_run_time)
        model_run_info += '{dt.year:d}-{dt.month:02d}-{dt.day:02d} '
        model_run_info += '{dt.hour:02d}{dt.minute:02d}Z'
        model_run_info = model_run_info.format(dt=mr_dtobj)

        selected_pt_info = 'No point selected'
        if self.selected_point is not None:
            if self.selected_point[0] > 0.0:
                lat_str = '{0:.2f} N'.format(abs(self.selected_point[0]))
            else:
                lat_str = '{0:.2f} S'.format(abs(self.selected_point[0]))

            if self.selected_point[1] > 0.0:
                long_str = '{0:.2f} E'.format(abs(self.selected_point[1]))
            else:
                long_str = '{0:.2f} W'.format(abs(self.selected_point[1]))


            sample_pts = [('latitude', self.selected_point[0]),
                          ('longitude', self.selected_point[1])]
            select_val_cube = \
                current_cube.interpolate(sample_pts,
                                            iris.analysis.Linear())

            select_val = float(select_val_cube.data)

            field_val_str = \
                'value: {val:.2f} {unit_str}'.format(val=select_val,
                                                     unit_str=unit_str)

            selected_pt_info = 'selected point {lat},{long}<br>'
            selected_pt_info += 'field value {fv}'
            selected_pt_info = selected_pt_info.format(lat=lat_str,
                                                       long=long_str,
                                                       fv=field_val_str)

        stats_str_list += [model_run_info,'']
        stats_str_list += [selected_pt_info, '']
        stats_str_list += ['Max = {0:.4f} {1}'.format(max_val, unit_str)]
        stats_str_list += ['Min = {0:.4f} {1}'.format(min_val, unit_str)]
        stats_str_list += ['Mean = {0:.4f} {1}'.format(mean_val, unit_str)]
        stats_str_list += ['STD = {0:.4f} {1}'.format(std_val, unit_str)]
        stats_str_list += ['RMS = {0:.4f} {1}'.format(rms_val, unit_str)]
        self.stats_string = '</br>'.join(stats_str_list)

    def update_title(self, current_cube):
        '''Update plot title.
        '''
        try:
            datestr1 = forest.util.get_time_str(self.current_time)
        except:
            datestr1 = self.current_time

        str1 = \
            '{plot_desc} {var_name} at {fcst_time}'.format(
                var_name=self.current_var,
                fcst_time=datestr1,
                plot_desc=self.plot_description,
                )
        self.current_title = \
            '\n'.join(textwrap.wrap(str1,
                                    ForestPlot.TITLE_TEXT_WIDTH))
    def create_plot(self):
        '''Main plotting function. Generic elements of the plot are created
        here, and then the plotting function for the specific variable is
        called using the self.plot_funcs dictionary.
        '''
        if self.visible:
            self.create_matplotlib_fig()
            self.create_bokeh_img_plot_from_fig()
        return self.bokeh_figure

    def create_matplotlib_fig(self):
        self.plot_funcs[self.current_var]()

    def _setup_tools(self):
        self.bokeh_tools = dict([(k1,None) for k1 in BOKEH_TOOLS_LIST])
        if 'pan' in BOKEH_TOOLS_LIST:
            self.bokeh_tools['pan'] = bokeh.models.PanTool()
        if 'wheel_zoom' in BOKEH_TOOLS_LIST:
            self.bokeh_tools['wheel_zoom'] = bokeh.models.WheelZoomTool()
        if 'reset' in BOKEH_TOOLS_LIST:
            self.bokeh_tools['reset'] = bokeh.models.ResetTool()
        if 'save' in BOKEH_TOOLS_LIST:
            self.bokeh_tools['save'] = bokeh.models.SaveTool()
        if 'box_zoom' in BOKEH_TOOLS_LIST:
            self.bokeh_tools['box_zoom'] = bokeh.models.BoxZoomTool()
        if 'hover' in BOKEH_TOOLS_LIST:
            self.bokeh_tools['hover'] = bokeh.models.HoverTool(
                tooltips=[
                    ("(x,y)", "($x, $y)"),
                ])

        self.active_bokeh_tools = {}
        self.active_bokeh_tools['drag'] = self.bokeh_tools['pan']
        self.active_bokeh_tools['inspect'] = self.bokeh_tools['hover']
        self.active_bokeh_tools['scroll'] = self.bokeh_tools['wheel_zoom']
        self.active_bokeh_tools['tap'] = None

    def create_bokeh_img_plot_from_fig(self):
        cur_region = self.region_dict[self.current_region]
        if self.current_img_array is not None:
            self.create_bokeh_img()
        else:
            mid_x = (cur_region[2] + cur_region[3]) * 0.5
            mid_y = (cur_region[0] + cur_region[1]) * 0.5
            self.bokeh_figure.text(x=[mid_x],
                                   y=[mid_y],
                                   text=['Plot loading'],
                                   text_color=['#FF0000'],
                                   text_font_size="20pt",
                                   text_baseline="middle",
                                   text_align="center",
                                   )

        # Add cartopy coastline to bokeh figure
        x_start = cur_region[2]
        x_end = cur_region[3]
        y_start = cur_region[0]
        y_end = cur_region[1]
        coastlines(self.bokeh_figure,
                   scale=self.coast_res,
                   extent=(x_start, x_end, y_start, y_end))

        self.bokeh_figure.title.text = self.current_title

    def create_bokeh_img(self):
        '''create bokeh image from settings or mappable'''
        x, y, dw, dh = self.get_x_y_dw_dh()
        self.bokeh_image = \
            self.bokeh_figure.image_rgba(image=[self.current_img_array],
                                         x=[x],
                                         y=[y],
                                         dw=[dw],
                                         dh=[dh])
        self.bokeh_img_ds = self.bokeh_image.data_source

    def get_x_y_dw_dh(self):
        image_source = 'mappable'
        if image_source == 'mappable':
            # Use mappable.get_extent() to define image_rgba()
            if hasattr(self.main_plot, 'get_extent'):
                left, right, bottom, top = self.main_plot.get_extent()
            else:
                left, right = self.coords_long.min(), self.coords_long.max()
                bottom, top = self.coords_lat.min(), self.coords_lat.max()
            x = left
            y = bottom
            dw = right - left
            dh = top - bottom
        else:
            # Use current region settings to define image_rgba()
            cur_region = self.region_dict[self.current_region]
            latitude_range = cur_region[1] - cur_region[0]
            longitude_range = cur_region[3] - cur_region[2]
            x = cur_region[2]
            y = cur_region[0]
            dw = longitude_range
            dh = latitude_range
        return x, y, dw, dh

    def update_bokeh_img_plot_from_fig(self):
        '''Update image_rgba() data source'''
        if self.bokeh_img_ds:
            x, y, dw, dh = self.get_x_y_dw_dh()
            self.bokeh_img_ds.data[u'image'] = [self.current_img_array]
            self.bokeh_img_ds.data[u'x'] = [x]
            self.bokeh_img_ds.data[u'y'] = [y]
            self.bokeh_img_ds.data[u'dw'] = [dw]
            self.bokeh_img_ds.data[u'dh'] = [dh]
        else:
            try:
                self.create_bokeh_img()
            except:
                self.current_img_array = None
        self.bokeh_figure.title.text = self.current_title

    def update_plot(self):
        print("{}.update_plot() called".format(self.__class__.__name__))

    def mpl_update_plot(self):
        '''Main plot update function. Generic elements of the plot are
        updated here where possible, and then the plot update function for
        the specific variable is called using the self.plot_funcs dictionary.
        '''
        self.update_funcs[self.current_var]()
        self.current_figure.canvas.draw_idle()
        self.update_bokeh_img_plot_from_fig()
        if self.stats_widget:
            self.update_stats_widget()

    def create_stats_widget(self):
        self.stats_widget = bokeh.models.widgets.Div(text=self.stats_string,
                                                     height=200,
                                                     width=400,
                                                     )
        return self.stats_widget

    def create_colorbar_widget(self):
        colorbar_html = "<img src='" + self.app_path + "/static/" + \
                        self.colorbar_link + "'\>"
        self.colorbar_widget = bokeh.models.widgets.Div(text=colorbar_html,
                                                        height=100,
                                                        width=800,
                                                        )
        return self.colorbar_widget

    def update_stats_widget(self):
        print('Updating stats widget')
        try:
            self.stats_widget.text = self.stats_string
        except AttributeError as e1:
            print('Unable to update stats as stats widget not initiated')

    def update_colorbar_widget(self):
        self.colorbar_link = self.current_var + '_colorbar.png'
        colorbar_html = "<img src='" + self.app_path + "/static/" + \
                        self.colorbar_link + "'\>"

        print(colorbar_html)

        try:
            self.colorbar_widget.text = colorbar_html
        except AttributeError as e1:
            print('Unable to update colorbar as colorbar widget not initiated')

    def set_data_time(self, new_time):
        print('selected new time {0}'.format(new_time))

        self.current_time = new_time
        # self.update_plot()
        if not self.visible:
            return
        self.create_matplotlib_fig()
        self.update_bokeh_img_plot_from_fig()
        if self.stats_widget:
            self.update_stats_widget()
        if self.colorbar_widget:
            self.update_colorbar_widget()

    def set_var(self, new_var):
        print('selected new var {0}'.format(new_var))
        self.current_var = new_var

        if self.visible:
            self.create_matplotlib_fig()
            self.update_bokeh_img_plot_from_fig()
            if self.stats_widget:
                self.update_stats_widget()
            if self.colorbar_widget:
                self.update_colorbar_widget()

    def _set_region(self, region):
        '''Event handler for a change in the selected plot region'''
        print('selected new region {0}'.format(region))
        self.current_region = region
        self.data_bounds = self.region_dict[self.current_region]
        self.create_matplotlib_fig()
        self.update_bokeh_img_plot_from_fig()
        if self.stats_widget:
            self.update_stats_widget()

    def set_region(self, region):
        """Adjust bokeh figure extents"""
        self.current_region = region
        extents = self.region_dict[self.current_region]
        y_start, y_end, x_start, x_end = extents
        self.bokeh_figure.x_range.start = x_start
        self.bokeh_figure.x_range.end = x_end
        self.bokeh_figure.y_range.start = y_start
        self.bokeh_figure.y_range.end = y_end

    def set_config(self, new_config):
        '''Function to set a new value of config and do an update
        '''
        print('setting new config {0}'.format(new_config))
        self._set_config_value(new_config)
        if self.visible:
            self.create_matplotlib_fig()
            self.update_bokeh_img_plot_from_fig()
            if self.stats_widget:
                self.update_stats_widget()

    def set_dataset(self, new_dataset, new_model_run_time):
        self.dataset = new_dataset
        self.model_run_time = new_model_run_time
        if self.visible:
            self.create_matplotlib_fig()
            self.update_bokeh_img_plot_from_fig()
            if self.stats_widget:
                self.update_stats_widget()

    def set_selected_point(self, latitude, longitude):
        self.selected_point = (latitude, longitude)
        if self.stats_widget:
            current_data = \
                self.get_data(self.stats_data_var[self.current_var])
            self.update_stats(current_data)
            self.update_stats_widget()


    def link_axes_to_other_plot(self, other_plot):
        try:
            self.bokeh_figure.x_range = other_plot.bokeh_figure.x_range
            self.bokeh_figure.y_range = other_plot.bokeh_figure.y_range
        except:
            print('bokeh plot linking failed.')


class ForestTimeSeries():
    """
    Class representing a time series plot. Unlike a map based plot, where
    each plot represents a single dataset, the timeseries plot plot several
    datasets together in the same plot for comparison purposes.
    """
    def __init__(self,
                 datasets,
                 model_run_time,
                 selected_point,
                 current_var):
        """

        :param datasets: A dictionary object of datasets
        :param model_run_time: A string representing the model run to be
                               displayed. All configs for the given model run
                               will be displayed on the timeseries graph.
        :param selected_point: The lat/long coordinates of the point to
                               display the timeseries for.
        :param current_var: The current variable to be displyed
                            e.g. precipitation
        """
        self.datasets = datasets
        self.model_run_time = model_run_time
        self.current_point = selected_point
        self.current_fig = None
        self.current_var = current_var
        self.cds_dict = {}

        self.placeholder_data = {'x_values': [0.0,1.0],
                                 'y_values': [0.0,0.0]}


    def __str__(self):
        """

        :return: A string describing the class.
        """
        return 'Class representing a time series plot in the forest tool'

    def create_plot(self):
        """
        Create the timeseries plot in a bokeh figure. This is where the actual
        work is done.
        :return: A bokeh figure object containing the timeseries plot.
        """
        self.current_fig = bokeh.plotting.figure(tools=BOKEH_TOOLS_LIST)
        self.bokeh_lines = {}
        self.cds_list = {}
        for ds_name in self.datasets.keys():
            current_ds = self.datasets[ds_name]['data']
            times1 = current_ds.get_times(self.current_var)
            times1 = times1 - times1[0]
            var_cube = \
                current_ds.get_timeseries(self.current_var,
                                          self.current_point)
            if var_cube:
                var_values = var_cube.data

                data1 = {'x_values': times1,
                         'y_values': var_values}

                ds_source = bokeh.models.ColumnDataSource(data=data1)

                ds_line_plot = self.current_fig.line(x='x_values',
                                                     y='y_values',
                                                     source=ds_source,
                                                     name=ds_name)
            else:
                ds_source = \
                    bokeh.models.ColumnDataSource(data=self.placeholder_data)
                ds_line_plot = self.current_fig.line(x='x_values',
                                                     y='y_values',
                                                     source=ds_source,
                                                     name=ds_name)

            self.cds_dict[ds_name] = ds_source
            self.bokeh_lines[ds_name] = ds_line_plot

        self._update_fig_title()

        return self.current_fig

    def _update_plot(self):
        """
        Update the bokeh figure with a new timeseries plot. This is called by
        functions that are called to change an input, and should not be called
        by a user directly.
        :return:
        """
        for ds_name in self.datasets.keys():
            if self.cds_dict[ds_name] is not None:
                current_ds = self.datasets[ds_name]['data']
                times1 = current_ds.get_times(self.current_var)
                times1 = times1 - times1[0]
                var_cube = \
                    current_ds.get_timeseries(self.current_var,
                                              self.current_point)

                if var_cube is not None:
                    var_values = var_cube.data

                    data1 = {'x_values': times1,
                         'y_values': var_values}

                    self.cds_dict[ds_name].data = data1
                else:
                    self.cds_dict[ds_name].data = self.placeholder_data
        self._update_fig_title()

    def _update_fig_title(self):
        """
        Update the title of the bokeh figure.
        :return: No return.
        """
        fig_title = 'Plotting variable {var} for model run {mr}'
        fig_title = fig_title.format(var=self.current_var,
                                     mr=str(self.model_run_time))
        self.current_fig.title.text = fig_title

    def set_var(self, new_var):
        """
        Set the timeseries to display the
        :param new_var: The new variable to be displayed.
        :return: No return value
        """
        self.current_var = new_var
        self._update_plot()

    def set_selected_point(self, latitude, longitude):
        """
        Set a new location for the timeseries display.
        :param latitude: Latitude of the location to display.
        :param longitude: longitude of the location to display.
        :return: No return value.
        """
        self.current_point = (latitude, longitude)
        self._update_plot()

    def set_data_time(self, new_time):
        """
        This function is provided for a consistent interface with other plot
        classes, but does nothing.
        """
        pass

    def set_dataset(self, new_dataset, new_model_run_time):
        """
        Set a new model run as the input to the timeseries.
        :param new_dataset: The dictionary representing the new model run
                            dataset.
        :param new_model_run_time: A string representing the model run time.
        :return: No return value.
        """
        self.datasets = new_dataset
        self.model_run_time = new_model_run_time
