import textwrap

import multiprocessing
import pdb

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


class ForestPlot(object):

    '''
    Main plot class. The plotting function is create_plot().
    '''
    TITLE_TEXT_WIDTH = 40
    PRESSURE_LEVELS_HPA = range(980, 1030, 2)

    DISPLAY_MODE_NONE = 'none'
    DISPLAY_MODE_PLOT = 'plot'
    DISPLAY_MODE_MISSING_DATA = 'missing_data'
    DISPLAY_MODE_LOADING = 'loading'

    def __init__(self,
                 dataset,
                 po1,
                 figname,
                 plot_var,
                 conf1,
                 reg1,
                 rd1,
                 unit_dict):
        '''
        Initialisation function for ForestPlot class
        '''
        self.region_dict = rd1
        self.current_time = None
        self.plot_options = po1
        self.dataset = dataset
        self.figure_name = figname
        self.current_var = plot_var
        self.current_config = conf1
        self.current_region = reg1
        self.current_time = \
            self.dataset[self.current_config]['data'].get_times(self.current_var)[0]
        self.data_bounds = self.region_dict[self.current_region]
        self.show_colorbar = False
        self.show_axis_ticks = False
        self.use_mpl_title = False
        self.current_title = ''
        self.stats_string = ''
        self.colorbar_link = plot_var + '_colorbar.png'
        self.bokeh_figure = None
        self.bokeh_image = None
        self.bokeh_img_ds = None
        self.unit_dict = unit_dict
        self.stats_widget = None
        self.colorbar_widget = None
        self.display_mode = ForestPlot.DISPLAY_MODE_NONE
        self.plot_msg_txt = ''
        self.plot_msg = None

        self.current_fig_size = (8.0, 6.0)
        self.bokeh_fig_width = 8
        self.coast_res = '50m'
        self.bokeh_fig_size = (800, 600)

    def __str__(self):
        return self.dataset[self.current_config]['data_type_name']

    def create_matplotlib_fig(self):
        print('create_matplotlib_fig start')
        self.mpl_plot = ForestMplPlot(self.dataset,
                                 self.plot_options,
                                 self.figure_name,
                                 self.current_var,
                                 self.current_config,
                                 self.current_region,
                                 self.region_dict,
                                 self.unit_dict,
                                 self.current_time,
                                 )

        print('created ForestMplPlot')

        self.mpl_plot.create_plot()
        self.stats_string = self.mpl_plot.stats_string
        self.current_title = self.mpl_plot.current_title
        self.current_img_array = self.mpl_plot.current_img_array

        self.display_mode = ForestPlot.DISPLAY_MODE_PLOT

    def _update_mpl_params(self):
        self.mpl_plot.current_time = self.current_time
        self.mpl_plot.current_var = self.current_var
        self.mpl_plot.current_config = self.current_config
        self.mpl_plot.current_region = self.current_region
        self.mpl_plot.data_bounds = self.data_bounds

    @forest.util.timer
    def create_plot(self):

        '''Main plotting function. Generic elements of the plot are created
        here, and then the plotting function for the specific variable is
        called using the self.plot_funcs dictionary.
        '''

        self.create_matplotlib_fig()
        self.create_bokeh_img_plot_from_fig()

        return self.bokeh_figure


    def create_bokeh_img_plot_from_fig(self):

        '''

        '''


        cur_region = self.region_dict[self.current_region]

        # Set figure navigation limits
        x_limits = bokeh.models.Range1d(cur_region[2], cur_region[3],
                                        bounds=(cur_region[2], cur_region[3]))
        y_limits = bokeh.models.Range1d(cur_region[0], cur_region[1],
                                        bounds=(cur_region[0], cur_region[1]))

        # Initialize figure
        self.bokeh_figure = \
            bokeh.plotting.figure(plot_width=self.bokeh_fig_size[0],
                                  plot_height=self.bokeh_fig_size[1],
                                  x_range=x_limits,
                                  y_range=y_limits,
                                  tools='pan,wheel_zoom,reset,save')

        self.create_bokeh_img()

        self.bokeh_figure.title.text = self.current_title

    def create_bokeh_img(self):

        '''

        '''

        cur_region = self.region_dict[self.current_region]
        # Add mpl image
        latitude_range = cur_region[1] - cur_region[0]
        longitude_range = cur_region[3] - cur_region[2]
        if self.display_mode == ForestPlot.DISPLAY_MODE_PLOT:
            self.plot_msg_txt = ''
            arr_to_plot = self.current_img_array
        elif self.display_mode == ForestPlot.DISPLAY_MODE_LOADING:
            self.plot_msg_txt = 'Plot loading'
            arr_to_plot = numpy.zeros((40,30,3),dtype='uint8')
        elif self.display_mode == ForestPlot.DISPLAY_MODE_MISSING_DATA:
            self.plot_msg_txt = 'Data missing'
            arr_to_plot = numpy.zeros((40, 30, 3), dtype='uint8')
        self.bokeh_image = \
            self.bokeh_figure.image_rgba(image=[self.current_img_array],
                                         x=[cur_region[2]],
                                         y=[cur_region[0]],
                                         dw=[longitude_range],
                                         dh=[latitude_range])
        self.bokeh_img_ds = self.bokeh_image.data_source

        cur_region = self.region_dict[self.current_region]
        mid_x = (cur_region[2] + cur_region[3]) * 0.5
        mid_y = (cur_region[0] + cur_region[1]) * 0.5
        self.plot_msg = \
            self.bokeh_figure.text(x=[mid_x],
                                   y=[mid_y],
                                   text=[self.plot_msg_txt],
                                   text_color=['#FF0000'],
                                   text_font_size="20pt",
                                   text_baseline="middle",
                                   text_align="center",
                                   )

    def update_bokeh_img_plot_from_fig(self):

        '''

        '''

        cur_region = self.region_dict[self.current_region]

        if self.bokeh_img_ds:
            self.bokeh_img_ds.data[u'image'] = [self.current_img_array]
            self.bokeh_img_ds.data[u'x'] = [cur_region[2]]
            self.bokeh_img_ds.data[u'y'] = [cur_region[0]]
            self.bokeh_img_ds.data[u'dw'] = [cur_region[3] - cur_region[2]]
            self.bokeh_img_ds.data[u'dh'] = [cur_region[1] - cur_region[0]]
            self.bokeh_figure.title.text = self.current_title

        else:
            try:
                self.create_bokeh_img()
                self.bokeh_figure.title.text = self.current_title
            except:
                self.current_img_array = None
        if self.plot_msg:
            if self.plot_msg_txt:
                self.plot_msg.glyph.text = self.plot_msg_txt
            else:
                self.plot_msg.visible = False

    def update_plot(self):

        '''Main plot update function. Generic elements of the plot are
        updated here where possible, and then the plot update function for
        the specific variable is called using the self.plot_funcs dictionary.
        '''
        if self.mpl_plot:
            print('updating plot')
            self._update_mpl_params()
            self.mpl_plot.update_plot()
            self.stats_string = self.mpl_plot.stats_string
            self.current_title = self.mpl_plot.current_title
            self.current_img_array = self.mpl_plot.current_img_array

            self.update_bokeh_img_plot_from_fig()
            if self.stats_widget:
                self.update_stats_widget()

    def create_stats_widget(self):

        '''

        '''

        self.stats_widget = bokeh.models.widgets.Div(text=self.stats_string,
                                                     height=400,
                                                     width=800,
                                                     )
        return self.stats_widget

    def create_colorbar_widget(self):

        '''

        '''



        colorbar_html = "<img src='plot_sea_two_model_comparison/static/" + \
                       self.colorbar_link + "'\>"

        self.colorbar_widget = bokeh.models.widgets.Div(text=colorbar_html,
                                                        height=100,
                                                        width=800,
                                                        )
        return self.colorbar_widget

    def update_stats_widget(self):

        '''

        '''


        try:
            self.stats_widget.text = self.stats_string
        except AttributeError as e1:
            print('Unable to update stats as stats widget not initiated')

    def update_colorbar_widget(self):

        '''

        '''

        self.colorbar_link = self.current_var + '_colorbar.png'
        colorbar_html = "<img src='plot_sea_two_model_comparison/static/" + \
                       self.colorbar_link + "'\>"


        try:
            self.colorbar_widget.text = colorbar_html
        except AttributeError as e1:
            print('Unable to update colorbar as colorbar widget not initiated')

    def set_data_time(self, new_time):

        '''

        '''

        print('selected new time {0}'.format(new_time))

        self.current_time = new_time
        self.update_plot()

    def set_var(self, new_var):

        '''

        '''

        print('selected new var {0}'.format(new_var))

        self.current_var = new_var
        self.create_matplotlib_fig()
        self.update_bokeh_img_plot_from_fig()
        if self.stats_widget:
            self.update_stats_widget()
        if self.colorbar_widget:
            self.update_colorbar_widget()

    def set_region(self, new_region):

        '''Event handler for a change in the selected plot region.

        '''

        print('selected new region {0}'.format(new_region))

        self.current_region = new_region
        self.data_bounds = self.region_dict[self.current_region]
        self.create_matplotlib_fig()
        self.update_bokeh_img_plot_from_fig()
        if self.stats_widget:
            self.update_stats_widget()

    def set_config(self, new_config):

        '''Function to set a new value of config and do an update

        '''

        print('setting new config {0}'.format(new_config))
        self.current_confog = new_config
        self.create_matplotlib_fig()
        self.update_bokeh_img_plot_from_fig()
        if self.stats_widget:
            self.update_stats_widget()

    def link_axes_to_other_plot(self, other_plot):

        '''

        '''

        try:
            self.bokeh_figure.x_range = other_plot.bokeh_figure.x_range
            self.bokeh_figure.y_range = other_plot.bokeh_figure.y_range
        except:
            print('bokeh plot linking failed.')


class ForestMplPlot(object):

    def __init__(self,
                 dataset,
                 po1,
                 figname,
                 plot_var,
                 conf1,
                 reg1,
                 rd1,
                 unit_dict,
                 current_time):
        # store input parameters in object
        self.region_dict = rd1
        self.current_time = current_time
        self.plot_options = po1
        self.dataset = dataset
        self.figure_name = figname
        self.current_var = plot_var
        self.current_config = conf1
        self.current_region = reg1
        self.data_bounds = self.region_dict[self.current_region]

        # set some default values
        self.current_fig_size = (8.0, 6.0)
        self.coast_res = '50m'


        # initialise plot related variables which will assigned in plot
        # creation process
        self.main_plot = None
        self.show_colorbar = False
        self.show_axis_ticks = False
        self.use_mpl_title = False
        self.current_title = ''
        self.stats_string = ''
        self.colorbar_link = plot_var + '_colorbar.png'
        self.unit_dict = unit_dict
        self.stats_widget = None
        self.colorbar_widget = None
        self.plot_msg_txt = ''
        self.plot_msg = None
        self.plot_description = ''

        # call some init functions
        self.setup_plot_funcs()
        self.setup_pressure_labels()

    def __str__(self):
        return self.dataset[self.current_config]['data_type_name']

    def setup_pressure_labels(self):
        '''
        Create dict of pressure levels, to be used labelling MSLP contour
        plots.
        '''
        self.mslp_contour_label_dict = {}
        for pressure1 in ForestPlot.PRESSURE_LEVELS_HPA:
            self.mslp_contour_label_dict[
                pressure1] = '{0:d}hPa'.format(int(pressure1))

    def setup_plot_funcs(self):
        '''
        Set up dictionary of plot functions. This is used by the main
        create_plot() function to call the plotting function relevant to the
        specific variable being plotted. There is also a second dictionary
        which is by the update_plot() function, which does the minimum amount
        of work to update the plot, and is used for some option changes,
        mainly a change in the forecast time selected.
        '''
        self.plot_funcs = {'precipitation': self.plot_precip,
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

    def create_plot(self):
        self.create_matplotlib_fig()

    def update_plot(self):
        self.update_funcs[self.current_var]()
        if self.use_mpl_title:
            self.current_axes.set_title(self.current_title)

        self.current_figure.set_figwidth(self.current_fig_size[0])
        cur_region = self.region_dict[self.current_region]
        self.current_figure.set_figheight(
            round(self.current_figure.get_figwidth() *
                  (cur_region[1] - cur_region[0]) /
                  (cur_region[3] - cur_region[2]), 2))
        self.current_figure.canvas.draw_idle()
        self.current_img_array = forest.util.get_image_array_from_figure(
            self.current_figure)


    def create_matplotlib_fig(self):
        '''

        '''
        self.current_figure = \
            matplotlib.pyplot.figure(self.figure_name,
                                     figsize=self.current_fig_size)
        self.current_figure.clf()
        self.current_axes = \
            self.current_figure.add_subplot(
                111,
                projection=cartopy.crs.PlateCarree())
        self.current_axes.set_position([0, 0, 1, 1])
        self.plot_funcs[self.current_var]()
        if self.main_plot:
            if self.use_mpl_title:
                self.current_axes.set_title(self.current_title)
            self.current_axes.set_xlim(
                self.data_bounds[2], self.data_bounds[3])
            self.current_axes.set_ylim(
                self.data_bounds[0], self.data_bounds[1])
            self.current_axes.xaxis.set_visible(self.show_axis_ticks)
            self.current_axes.yaxis.set_visible(self.show_axis_ticks)
            if self.show_colorbar:
                self.current_figure.colorbar(self.main_plot,
                                             orientation='horizontal')

            self.current_figure.canvas.draw()

        self.current_img_array = forest.util.get_image_array_from_figure(
            self.current_figure)

    def create_blank(self):
        self.current_title = 'Blank plot'
        self.stats_string = ''

    def get_data(self, var_name=None):
        current_ds = self.dataset[self.current_config]['data']
        var_to_load = var_name
        if not var_to_load:
            var_to_load = self.current_var
        data_cube = current_ds.get_data(var_to_load,
                                        convert_units=True,
                                        selected_time=self.current_time)
        return data_cube

    def update_coords(self, data_cube):
        '''
        Update the latitude and longitude coordinates for the data.
        '''

        self.coords_lat = data_cube.coords('latitude')[0].points
        self.coords_long = data_cube.coords('longitude')[0].points

    def update_precip(self):
        '''
        Update function for precipitation plots, called by update_plot() when
        precipitation is the selected plot type.
        '''
        print('updating precip')
        data_cube = self.get_data()
        array_for_update = data_cube.data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(data_cube)
        self.update_stats(data_cube)

    def plot_precip(self):
        '''
        Function for creating precipitation plots, called by create_plot when
        precipitation is the selected plot type.
        '''

        data_cube = self.get_data()
        self.update_coords(data_cube)
        self.current_axes.coastlines(resolution='110m')
        self.current_axes.coastlines(resolution='110m')
        self.main_plot = \
            self.current_axes.pcolormesh(self.coords_long,
                                         self.coords_lat,
                                         data_cube.data,
                                         cmap=self.plot_options[
                                             self.current_var]['cmap'],
                                         norm=self.plot_options[
                                             self.current_var]['norm'],
                                         transform=cartopy.crs.PlateCarree())
        self.update_title(data_cube)
        self.update_stats(data_cube)

    def update_wind_vectors(self):
        '''
        Update function for wind vector plots, called by update_plot() when
        wind vectors is the selected plot type.
        '''

        wind_speed_cube = self.get_data('wind_speed')
        array_for_update = wind_speed_cube.data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(wind_speed_cube)
        self.update_stats(wind_speed_cube)
        wv_u_data = self.get_data('wv_U')
        wv_v_data = self.get_data('wv_V')
        self.quiver_plot.set_UVC(wv_u_data,
                                 wv_v_data)

    def plot_wind_vectors(self):
        '''
        Function for creating wind vector plots, called by create_plot when
        wind vectors is the selected plot type.
        '''

        wind_speed_cube = self.get_data('wind_speed')
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

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical',
                                                            'coastline',
                                                            self.coast_res,
                                                            edgecolor='0.5',
                                                            facecolor='none')
        self.current_axes.add_feature(coastline_50m)

        self.quiver_plot = \
            self.current_axes.quiver(
                self.get_data('wv_X'),
                self.get_data('wv_Y'),
                self.get_data('wv_U'),
                self.get_data('wv_V'),
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
        '''
        Update function for wind speed with MSLP contours plots, called by
        update_plot() when wind speed with MSLP is the selected plot type.
        '''
        wind_speed_cube = self.get_data('wind_speed')

        array_for_update = wind_speed_cube.data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        # to update contours, remove old elements and generate new contours
        for c1 in self.mslp_contour.collections:
            self.current_axes.collections.remove(c1)

        ap_cube = self.get_data('mslp')
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
        '''
        Function for creating wind speed with MSLP contour plots, called by
        create_plot when wind speed with MSLP contours is the selected plot
        type.
        '''
        wind_speed_cube = self.get_data('wind_speed')
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

        ap_cube = self.get_data('mslp')
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

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical',
                                                            'coastline',
                                                            self.coast_res,
                                                            edgecolor='0.5',
                                                            facecolor='none')

        self.current_axes.add_feature(coastline_50m)
        self.update_title(wind_speed_cube)

    def update_wind_streams(self):
        '''
        Update function for wind streamline plots, called by update_plot()
        when wind streamlines is the selected plot type.
        '''
        wind_speed_cube = self.get_data('wind_speed')
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
                self.get_data('wv_X_grid'),
                self.get_data('wv_Y_grid'),
                self.get_data('wv_U'),
                self.get_data('wv_V'),
                color='k',
                density=[0.5, 1.0])
        # we need to manually keep track of arrows so they can be removed when
        # the plot is updated
        pl2 = list(self.current_axes.patches)
        self.wind_stream_patches = [p1 for p1 in pl2 if p1 not in pl1]

    def plot_wind_streams(self):
        '''
        Function for creating wind streamline plots, called by create_plot when
        wind streamlines is the selected plot type.
        '''
        n=0
        wind_speed_cube = self.get_data('wind_speed')
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
                self.get_data('wv_X_grid'),
                self.get_data('wv_Y_grid'),
                self.get_data('wv_U'),
                self.get_data('wv_V'),
                color='k',
                density=[0.5, 1.0])

        # we need to manually keep track of arrows so they can be removed when
        # the plot is updated
        pl2 = list(self.current_axes.patches)
        self.wind_stream_patches = [p1 for p1 in pl2 if p1 not in pl1]

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical',
                                                            'coastline',
                                                            self.coast_res,
                                                            edgecolor='0.5',
                                                            facecolor='none')
        self.current_axes.add_feature(coastline_50m)
        self.update_title(wind_speed_cube)

    def update_air_temp(self):
        '''
        Update function for air temperature plots, called by update_plot() when
        air temperature is the selected plot type.
        '''
        at_cube = self.get_data(self.current_var)
        array_for_update = at_cube.data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(at_cube)
        self.update_stats(at_cube)

    def plot_air_temp(self):
        '''
        Function for creating air temperature plots, called by create_plot when
        air temperature is the selected plot type.
        '''
        print('start plot air temp')

        at_cube = self.get_data(self.current_var)
        print('plot air temp 01')
        self.update_coords(at_cube)
        print('plot air temp 02')
        self.main_plot = \
            self.current_axes.pcolormesh(self.coords_long,
                                         self.coords_lat,
                                         at_cube.data,
                                         cmap=self.plot_options[
                                             self.current_var]['cmap'],
                                         norm=self.plot_options[
                                             self.current_var]['norm']
                                         )

        # Add coastlines to the map created by contourf.
        print('plot air temp 03')
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical',
                                                            'coastline',
                                                            self.coast_res,
                                                            edgecolor='0.5',
                                                            facecolor='none')
        self.current_axes.add_feature(coastline_50m)
        print('plot air temp 04')
        self.update_title(at_cube)
        print('plot air temp 05')
        self.update_stats(at_cube)
        print('end plot air temp')

    def update_mslp(self):
        '''
        Update function for MSLP plots, called by update_plot() when
        MSLP is the selected plot type.
        '''
        ap_cube = self.get_data(self.current_var)
        array_for_update = ap_cube.data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(ap_cube)
        self.update_stats(ap_cube)

    def plot_mslp(self):
        '''
        Function for creating MSLP plots, called by create_plot when
        MSLP is the selected plot type.
        '''
        ap_cube = self.get_data(self.current_var)
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

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical',
                                                            'coastline',
                                                            self.coast_res,
                                                            edgecolor='0.5',
                                                            facecolor='none')
        self.current_axes.add_feature(coastline_50m)
        self.update_title(ap_cube)
        self.update_stats(ap_cube)

    def update_cloud(self):
        '''
        Update function for cloud fraction plots, called by update_plot() when
        cloud fraction is the selected plot type.
        '''
        cloud_cube = self.get_data(self.current_var)
        array_for_update = cloud_cube.data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(cloud_cube)
        self.update_stats(cloud_cube)

    def plot_cloud(self):
        '''
        Function for creating cloud fraction plots, called by create_plot when
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

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical',
                                                            'coastline',
                                                            self.coast_res,
                                                            edgecolor='0.5',
                                                            facecolor='none')
        self.current_axes.add_feature(coastline_50m)
        self.update_title(cloud_cube)
        self.update_stats(cloud_cube)

    def update_him8(self):
        '''
        Update function for himawari-8 image plots, called by update_plot()
        when cloud fraction is the selected plot type.
        '''

        him8_image = self.get_data('himawari-8')
        self.current_axes.images.remove(self.main_plot)
        self.main_plot = self.current_axes.imshow(him8_image,
                                                  extent=(self.data_bounds[2],
                                                          self.data_bounds[3],
                                                          self.data_bounds[0],
                                                          self.data_bounds[1]),
                                                  origin='upper')
        self.update_title(None)

    def plot_him8(self):
        '''
        Function for creating himawari-8 image plots, called by create_plot()
        when cloud fraction is the selected plot type.
        '''
        him8_image = self.get_data('himawari-8')
        self.main_plot = self.current_axes.imshow(him8_image,
                                                  extent=(self.data_bounds[2],
                                                          self.data_bounds[3],
                                                          self.data_bounds[0],
                                                          self.data_bounds[1]),
                                                  origin='upper')

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical',
                                                            'coastline',
                                                            self.coast_res,
                                                            edgecolor='g',
                                                            alpha=0.5,
                                                            facecolor='none')

        self.current_axes.add_feature(coastline_50m)
        self.current_axes.set_extent((self.data_bounds[2],
                                      self.data_bounds[3],
                                      self.data_bounds[0],
                                      self.data_bounds[1]))

        self.update_title(None)

    def update_simim(self):
        '''
        Update function for himawari-8 image plots, called by update_plot() when
        cloud fraction is the selected plot type.
        '''
        simim_cube = self.get_data('simim')
        array_for_update = simim_cube.data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(None)

    def plot_simim(self):
        '''
        Function for creating himawari-8 image plots, called by create_plot()
        when cloud fraction is the selected plot type.
        '''
        simim_cube = self.get_data('simim')
        lats = simim_cube.coords('grid_latitude')[0].points
        lons = simim_cube.coords('grid_longitude')[0].points
        self.main_plot = \
            self.current_axes.pcolormesh(lons,
                                         lats,
                                         simim_cube.data,
                                         cmap=self.plot_options[
                                             self.current_var]['cmap'],
                                         norm=self.plot_options[
                                             self.current_var]['norm']
                                         )

        # Add coastlines to the map created by contourf
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical',
                                                            'coastline',
                                                            self.coast_res,
                                                            edgecolor='g',
                                                            alpha=0.5,
                                                            facecolor='none')

        self.current_axes.add_feature(coastline_50m)
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
        # pdb.set_trace()
        stats_str_list = [self.current_title]
        unit_str = self.unit_dict[self.current_var]
        data_at_time = current_cube
        max_val = numpy.max(data_at_time.data)
        min_val = numpy.min(data_at_time.data)
        mean_val = numpy.mean(data_at_time.data)
        std_val = numpy.std(data_at_time.data)
        rms_val = numpy.sqrt(numpy.mean(numpy.power(data_at_time.data, 2.0)))

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