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
import forest.geography

import numpy as np
import scipy.ndimage
import iris.analysis
import skimage.transform
from functools import lru_cache


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


BOKEH_TOOLS_LIST = ['pan', 'wheel_zoom', 'reset', 'save', 'box_zoom', 'hover']


class MissingDataError(Exception):
    def __init__(self, config, var, time):
        self.config = config
        self.var = var
        self.time = time


def add_x_axes(figure, position='above'):
    """Extra x-axis above figure"""
    figure.extra_x_ranges[position] = figure.x_range
    axis = bokeh.models.LinearAxis(x_range_name=position)
    axis.major_label_text_font_size = '0pt'
    figure.add_layout(axis, position)


def add_y_axes(figure, position="right"):
    """Extra y-axis right of figure"""
    figure.extra_y_ranges[position] = figure.y_range
    axis = bokeh.models.LinearAxis(y_range_name=position)
    axis.major_label_text_font_size = '0pt'
    figure.add_layout(axis, position)


def add_coastlines(bokeh_figure, extent):
    """Add coastlines to bokeh figure"""
    xs, ys = forest.geography.coastlines(extent)
    bokeh_figure.multi_line(xs, ys, color='black')


def add_borders(bokeh_figure, extent):
    """Add borders to bokeh figure"""
    xs, ys = forest.geography.borders(extent)
    bokeh_figure.multi_line(xs, ys, color='grey')


@forest.util.timer
def smooth_image(array, output_shape):
    """Smooth high resolution imagery"""
    use_scikit_image = True
    if use_scikit_image:
        resized = skimage.transform.resize(array,
                                           output_shape,
                                           mode='reflect')
        resized = 255 * resized
        return np.ascontiguousarray(resized, dtype=np.uint8)
    else:
        # My own implementation of skimage.transform.resize
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
                 app_path,
                 init_time,
                 bokeh_figure=None,
                 visible=True):
        '''Initialisation function for ForestPlot class
        '''
        self.current_figure = matplotlib.pyplot.figure(figure_name)
        self.current_axes = self.current_figure.add_subplot(111)

        self.region_dict = rd1
        self.main_plot = None
        self.current_time = init_time
        self.plot_options = plot_options
        self.dataset = dataset
        self.model_run_time = model_run_time
        self.current_var = plot_var
        self.current_config = conf1
        self.plot_description = self.dataset[
            self.current_config]['data_type_name']
        self.current_region = reg1
        self.app_path = app_path
        self.data_bounds = self.region_dict[self.current_region]
        self.plot_funcs = {'precipitation': self.plot_pcolormesh,
                           'accum_precip_3hr': self.plot_pcolormesh,
                           'accum_precip_6hr': self.plot_pcolormesh,
                           'accum_precip_12hr': self.plot_pcolormesh,
                           'accum_precip_24hr': self.plot_pcolormesh,
                           'wind_vectors': self.plot_wind_vectors,
                           'wind_mslp': self.plot_wind_mslp,
                           'wind_streams': self.plot_wind_streams,
                           'mslp': self.plot_pcolormesh,
                           'air_temperature': self.plot_pcolormesh,
                           'cloud_fraction': self.plot_pcolormesh,
                           'himawari-8': self.plot_him8,
                           'simim': self.plot_simim,
                           'W': self.plot_sat_simim_imagery,
                           'I': self.plot_sat_simim_imagery,
                           'V': self.plot_sat_simim_imagery,
                           'blank': self.create_blank,
                           }
        self.update_funcs = {'wind_vectors': self.update_wind_vectors,
                             'wind_mslp': self.update_wind_mslp,
                             'wind_streams': self.update_wind_streams,
                             'himawari-8': self.update_him8,
                             'simim': self.update_simim,
                             'W': self.update_sat_simim_imagery,
                             'I': self.update_sat_simim_imagery,
                             'V': self.update_sat_simim_imagery,
                             'blank': self.create_blank,
                             }
        self.mslp_contour_label_dict = {}
        for pressure1 in ForestPlot.PRESSURE_LEVELS_HPA:
            self.mslp_contour_label_dict[
                pressure1] = '{0:d}hPa'.format(int(pressure1))
        self.current_title = ''
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
        # IDEA:
        self.bokeh_image = self.bokeh_figure.image_rgba(image=[],
                                                        x=[],
                                                        y=[],
                                                        dw=[],
                                                        dh=[],
                                                        level="underlay")
        self.bokeh_img_ds = self.bokeh_image.data_source

        self.colorbar_widget = None
        self.visible = visible
        self._shape2d = None

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        if getattr(self, 'bokeh_image', None) is not None:
            self.bokeh_image.glyph.global_alpha = 1 if value else 0
        self._visible = value
        self.render()

    @forest.util.counter
    def create_plot(self):
        '''Main plotting function. Generic elements of the plot are created
        here, and then the plotting function for the specific variable is
        called using the self.plot_funcs dictionary.
        '''
        self.render()
        return self.bokeh_figure

    @forest.util.timer
    def render(self):
        """Plot RGBA images"""
        if not self.visible:
            return
        x, y, dw, dh, image = self.render_image(self.current_config,
                                                self.current_var,
                                                self.current_time)

        self.bokeh_img_ds.data.update({
            'image': [image],
            'x': [x],
            'y': [y],
            'dw': [dw],
            'dh': [dh]
        })
        self.bokeh_figure.title.text = self.current_title

    @lru_cache(maxsize=32)
    def render_image(self,
                     current_config,
                     current_var,
                     current_time):
        """Plot RGBA images"""
        self.plot_funcs[current_var]()

        # HACK: self._shape2d is populated by self.get_data()
        ni, nj = self._shape2d
        shape = (ni - 1, nj - 1)
        image = rgba_from_mappable(self.main_plot, shape)

        # Smooth high resolution imagery
        max_ni, max_nj = 800, 600
        ni, nj, _ = image.shape
        if (ni > max_ni) or (nj > max_nj):
            image = smooth_image(image, (max_ni, max_nj))

        # Plot bokeh image rgba
        x, y, dw, dh = self.get_x_y_dw_dh()
        return x, y, dw, dh, image

    def update_coords(self, data_cube):
        '''Update the latitude and longitude coordinates for the data.
        '''
        self.coords_lat = data_cube.coords('latitude')[0].points
        self.coords_long = data_cube.coords('longitude')[0].points

    def create_blank(self):
        self.main_plot = None
        self.current_title = 'Blank plot'

    @forest.util.timer
    def get_data(self, var_name, selected_time=None, config_name=None):
        if selected_time is None:
            selected_time = self.current_time
        if config_name is None:
            config_name = self.current_config
        config_data = self.dataset[config_name]['data']
        data_cube = config_data.get_data(var_name=var_name,
                                         selected_time=selected_time)
        # HACK: cache image array shape after get_data()
        self._shape2d = data_cube.data.shape
        return data_cube

    @forest.util.timer
    def plot_pcolormesh(self):
        cube = self.get_data(self.current_var)
        cmap = self.plot_options[self.current_var]['cmap']
        norm = self.plot_options[self.current_var]['norm']
        self.update_coords(cube)
        self.main_plot = \
            self.current_axes.pcolormesh(self.coords_long,
                                         self.coords_lat,
                                         cube.data,
                                         cmap=cmap,
                                         norm=norm,
                                         edgecolor='face')
        self.update_title(cube)

    def update_wind_vectors(self):
        '''Update function for wind vector plots, called by update_plot() when
        wind vectors is the selected plot type.
        '''
        wind_speed_cube = self.get_data(forest.data.WIND_SPEED_NAME)

        array_for_update = wind_speed_cube.data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(wind_speed_cube)
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

    def create_colorbar_widget(self):
        colorbar_html = "<img src='" + self.app_path + "/static/" + \
                        self.colorbar_link + "'\>"
        self.colorbar_widget = bokeh.models.widgets.Div(text=colorbar_html,
                                                        height=100,
                                                        width=800,
                                                        )
        return self.colorbar_widget

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
        if self.visible:
            self.render()
            if self.colorbar_widget:
                self.update_colorbar_widget()

    def set_var(self, new_var):
        print('selected new var {0}'.format(new_var))
        self.current_var = new_var
        if self.visible:
            self.render()
            if self.colorbar_widget:
                self.update_colorbar_widget()

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
        self.current_config = new_config
        self.plot_description = self.dataset[
            self.current_config]['data_type_name']
        if self.visible:
            self.render()

    def set_dataset(self, new_dataset, new_model_run_time):
        self.dataset = new_dataset
        self.model_run_time = new_model_run_time
        if self.visible:
            self.render()

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

        self.placeholder_data = {'x_values': [0.0, 1.0],
                                 'y_values': [0.0, 0.0]}

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
