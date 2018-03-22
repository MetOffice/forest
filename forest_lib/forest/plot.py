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


BOKEH_TOOLS_LIST = ['pan','wheel_zoom','reset','save','box_zoom']

class MissingDataError(Exception):
    def __init__(self, config, var, time):
        self.config = config
        self.var = var
        self.time = time


class ForestPlot(object):

    '''
    Main plot class. The plotting function is create_plot().
    '''
    TITLE_TEXT_WIDTH = 40
    PRESSURE_LEVELS_HPA = range(980, 1030, 2)

    MODE_PLOT = 'plot'
    MODE_LOADING = 'loading'
    MODE_MISSING_DATA = 'missing_data'

    def __init__(self,
                 dataset,
                 model_run_time,
                 po1,
                 figname,
                 plot_var,
                 conf1,
                 reg1,
                 rd1,
                 unit_dict,
                 unit_dict_display,
                 app_path,
                 init_time):

        '''Initialisation function for ForestPlot class
        '''

        self.region_dict = rd1
        self.main_plot = None
        self.current_time = init_time
        self.plot_options = po1
        self.dataset = dataset
        self.model_run_time = model_run_time
        self.figure_name = figname
        self.current_var = plot_var
        self._set_config_value(conf1)
        self.current_region = reg1
        self.app_path = app_path
        self.data_bounds = self.region_dict[self.current_region]
        self.show_colorbar = False
        self.show_axis_ticks = False
        self.use_mpl_title = False
        self.setup_plot_funcs()
        self.setup_pressure_labels()
        self.current_title = ''
        self.stats_string = ''
        self.colorbar_link = plot_var + '_colorbar.png'
        self.bokeh_figure = None
        self.bokeh_image = None
        self.bokeh_img_ds = None
        self.async = False
        self.unit_dict = unit_dict
        self.unit_dict_display = unit_dict_display
        self.stats_widget = None
        self.colorbar_widget = None

        self.current_figsize = (8.0, 6.0)
        self.bokeh_fig_size = (800,600)
        self.coast_res = '50m'
        self.display_mode = ForestPlot.MODE_LOADING



    def _set_config_value(self, new_config):

        '''

        '''

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

    def update_coords(self, data_cube):

        '''Update the latitude and longitude coordinates for the data.

        '''

        self.coords_lat = data_cube.coords('latitude')[0].points
        self.coords_long = data_cube.coords('longitude')[0].points

    def create_blank(self):

        '''

        '''

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
        self.update_coords(data_cube)
        self.current_axes.coastlines(resolution=self.coast_res)
        self.main_plot = \
            self.current_axes.pcolormesh(self.coords_long,
                                         self.coords_lat,
                                         data_cube.data,
                                         cmap=self.plot_options[
                                             self.current_var]['cmap'],
                                         norm=self.plot_options[
                                             self.current_var]['norm'],
                                         edgecolor='face',
                                         transform=cartopy.crs.PlateCarree())
        self.update_title(data_cube)
        self.update_stats(data_cube)

    def update_wind_vectors(self):

        '''Update function for wind vector plots, called by update_plot() when
        wind vectors is the selected plot type.

        '''

        wind_speed_cube = self.get_data()

        array_for_update = wind_speed_cube.data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(wind_speed_cube)
        self.update_stats(wind_speed_cube)
        wv_u_data = self.get_data(var_name='wv_U')
        wv_v_data = self.get_data(var_name='wv_V')
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

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical',
                                                            'coastline',
                                                            self.coast_res,
                                                            edgecolor='0.5',
                                                            facecolor='none')

        self.current_axes.add_feature(coastline_50m)
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
                self.get_data(var_name='wv_X_grid'),
                self.get_data(var_name='wv_Y_grid'),
                self.get_data(var_name='wv_U'),
                self.get_data(var_name='wv_V'),
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
                                         wind_speed_cube[
                                             self.current_time].data,
                                         cmap=self.plot_options[
                                             self.current_var]['cmap'],
                                         norm=self.plot_options[
                                             self.current_var]['norm']
                                         )
        pl1 = list(self.current_axes.patches)

        self.wind_stream_plot = \
            self.current_axes.streamplot(
                self.get_data(var_name='wv_X_grid'),
                self.get_data(var_name='wv_Y_grid'),
                self.get_data(var_name='wv_U'),
                self.get_data(var_name='wv_V'),
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

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical',
                                                            'coastline',
                                                            self.coast_res,
                                                            edgecolor='0.5',
                                                            facecolor='none')
        self.current_axes.add_feature(coastline_50m)
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

        '''

        '''

        if self.current_config == 'himawari-8':
            self.update_him8()
        elif self.current_config == 'simim':
            self.update_simim()

    def plot_sat_simim_imagery(self):

        '''

        '''

        if self.current_config == 'himawari-8':
            self.plot_him8()
        elif self.current_config == 'simim':
            self.plot_simim()

    def update_stats(self, current_cube):

        '''

        '''
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

        stats_str_list += [model_run_info,'']
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
    @forest.util.timer
    def create_plot(self):

        '''Main plotting function. Generic elements of the plot are created
        here, and then the plotting function for the specific variable is
        called using the self.plot_funcs dictionary.
        
        '''

        self.create_matplotlib_fig()
        self.create_bokeh_img_plot_from_fig()

        return self.bokeh_figure

    def create_matplotlib_fig(self):

        '''

        '''
        self.current_figure = matplotlib.pyplot.figure(self.figure_name,
                                                       figsize=self.current_figsize)
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

    def create_bokeh_img_plot_from_fig(self):

        '''

        '''

        self.current_img_array = forest.util.get_image_array_from_figure(
            self.current_figure)

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
                                  tools=','.join(BOKEH_TOOLS_LIST))

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

        self.bokeh_figure.title.text = self.current_title

    def create_bokeh_img(self):

        '''

        '''

        cur_region = self.region_dict[self.current_region]
        # Add mpl image
        latitude_range = cur_region[1] - cur_region[0]
        longitude_range = cur_region[3] - cur_region[2]
        self.bokeh_image = \
            self.bokeh_figure.image_rgba(image=[self.current_img_array],
                                         x=[cur_region[2]],
                                         y=[cur_region[0]],
                                         dw=[longitude_range],
                                         dh=[latitude_range])
        self.bokeh_img_ds = self.bokeh_image.data_source

    def update_bokeh_img_plot_from_fig(self):

        '''

        '''

        cur_region = self.region_dict[self.current_region]
        self.current_figure.set_figwidth(self.current_figsize[0])
        self.current_figure.set_figheight(
            round(self.current_figure.get_figwidth() *
                  (cur_region[1] - cur_region[0]) /
                  (cur_region[3] - cur_region[2]), 2))

        if self.bokeh_img_ds:
            self.current_img_array = forest.util.get_image_array_from_figure(
                self.current_figure)
            self.bokeh_img_ds.data[u'image'] = [self.current_img_array]
            self.bokeh_img_ds.data[u'x'] = [cur_region[2]]
            self.bokeh_img_ds.data[u'y'] = [cur_region[0]]
            self.bokeh_img_ds.data[u'dw'] = [cur_region[3] - cur_region[2]]
            self.bokeh_img_ds.data[u'dh'] = [cur_region[1] - cur_region[0]]
            self.bokeh_figure.title.text = self.current_title

        else:
            try:
                self.current_img_array = \
                    forest.util.get_image_array_from_figure(
                        self.current_figure)
                self.create_bokeh_img()
                self.bokeh_figure.title.text = self.current_title
            except:
                self.current_img_array = None

    def update_plot(self):

        '''Main plot update function. Generic elements of the plot are
        updated here where possible, and then the plot update function for
        the specific variable is called using the self.plot_funcs dictionary.
        
        '''

        self.update_funcs[self.current_var]()
        if self.use_mpl_title:
            self.current_axes.set_title(self.current_title)
        self.current_figure.canvas.draw_idle()
        if not self.async:
            self.update_bokeh_img_plot_from_fig()
        if self.stats_widget:
            self.update_stats_widget()

    def create_stats_widget(self):

        '''

        '''

        self.stats_widget = bokeh.models.widgets.Div(text=self.stats_string,
                                                     height=200,
                                                     width=400,
                                                     )
        return self.stats_widget

    def create_colorbar_widget(self):

        '''

        '''

        colorbar_html = "<img src='" + self.app_path + "/static/" + \
                        self.colorbar_link + "'\>"
            
        self.colorbar_widget = bokeh.models.widgets.Div(text=colorbar_html,
                                                        height=100,
                                                        width=800,
                                                        )
        return self.colorbar_widget

    def update_stats_widget(self):

        '''

        '''

        print('Updating stats widget')

        try:
            self.stats_widget.text = self.stats_string
        except AttributeError as e1:
            print('Unable to update stats as stats widget not initiated')

    def update_colorbar_widget(self):

        '''

        '''

        self.colorbar_link = self.current_var + '_colorbar.png'
        colorbar_html = "<img src='" + self.app_path + "/static/" + \
                        self.colorbar_link + "'\>"

        print(colorbar_html)

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
        if not self.async:
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
        if not self.async:
            self.update_bokeh_img_plot_from_fig()
            if self.stats_widget:
                self.update_stats_widget()

    def set_config(self, new_config):

        '''Function to set a new value of config and do an update

        '''

        print('setting new config {0}'.format(new_config))
        self._set_config_value(new_config)
        self.create_matplotlib_fig()
        if not self.async:
            self.update_bokeh_img_plot_from_fig()
            if self.stats_widget:
                self.update_stats_widget()

    def set_dataset(self, new_dataset, new_model_run_time):
        self.dataset = new_dataset
        self.model_run_time = new_model_run_time
        self.create_matplotlib_fig()
        if not self.async:
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