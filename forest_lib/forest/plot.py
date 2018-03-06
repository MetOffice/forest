"""Module containing a class to manage the plots for Forest. 

Functions
---------

None

Classes
-------

- ForestPlot -- Main class for containing Forest plots.

"""

import textwrap
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


class ForestPlot(object):

    """Main plot class. The plotting function is create_plot().
    
    Methods
    -------
    
    - __init__() -- Factory method.
    - _set_config_value() -- Set data config attributes.
    - setup_pressure_labels() -- Generate MSLP contour labels.
    - setup_plot_funcs() -- Assign plot functions to dictionary.
    - update_coords() -- Update coordinate attributes from cube.
    - create_blank() -- Creaet blank plot.
    - update_precip() -- Update precipitation plot.
    - plot_precip() -- Plot precipitation data.
    - update_wind_vectors() -- Update wind vector plot.
    - plot_wind_vectors() -- Plot wind vector data.
    - update_wind_mslp() -- Update wind and MSLP plot.
    - plot_wind_mslp() -- Plot wind and MSLP data.
    - update_wind_streams() -- Update wind stream plot.
    - plot_wind_streams() -- Plot wind stream data.
    - update_air_temp() -- Update air temperature plot.
    - plot_air_temp() -- Plot air temperature data.
    - update_mslp() -- Update MSLP plot.
    - plot_mslp() -- Plot MSLP data.
    - update_cloud() -- Update cloud fraction plot.
    - plot_cloud() -- Plot cloud fraction data.
    - update_him8() -- Update Himawari-8 plot.
    - plot_him8() -- Plot Himawari-8 image data.
    - update_simim() -- Update simulated imagery plot.
    - plot_simim() -- Plot simulated imagery data.
    - update_sat_simim_imagery() -- Call simim/him-8 update method.
    - plot_sat_simim_imagery() -- Call simim/him-8 plot method.
    - update_stats() -- Update data in stats widget.
    - update_title() -- Update plot title.
    - create_plot() -- Calls methods for creating figures.
    - create_matplotlib_fig() -- Create matplotlib figure.
    - create_bokeh_fig() -- Create Bokeh figure.
    - create_bokeh_img() -- Create Bokeh image data.
    - update_bokeh_img() -- Update image data in Bokeh plots.
    - update_plot() -- Update Bokeh plots.
    - create_stats_widget() -- Create Bokeh Div widget stats.
    - create_colorbar_widget() -- Create Bokeh Div widget colorbar.
    - update_stats_widget() -- Update Bokeh Div widget stats.
    - update_colorbar_widget() -- Update Bokeh Div widget colorbar.
    - set_data_time() -- Set plot data time.
    - set_var() -- Set plot variable.
    - set_region() -- Set plot region.
    - set_config() -- Set plot config.
    - link_axes_to_other_plot() -- Link Bokeh figure axes.
    
    Attributes
    ----------
    
    - region_dict -- Dict; Dict of regions and their lat/lon bounds.
    - main_plot -- matplotlib.pyplot plot; Curent plot object.
    - current_time -- Int; Current data time step.
    - plot_options -- Dict; Dict of colormaps and normalisations.
    - dataset -- ForestDataset; Dataset containing data to plot.
    - figure_name -- Str; Used to set matplotlib figure name.
    - current_var -- Str; Name of current plot variable.
    - current_region -- Str; Name of current plot region.
    - app_path -- Str; Path to app. Needed to find static/ dir.
    - data_bounds -- List; Lat/lon limits of plot region.
    - show_axis_ticks -- Bool; Include ticks in matplotlib plot.
    - use_mpl_title -- Bool; Include title in matplotlib plot.
    - current_title -- Str; Current title.
    - stats_string -- Str; HTML used for stats string.
    - colorbar_link -- Str; Local path to colorbar PNG.
    - bokeh_figure -- bokeh.plotting.figure; Bokeh figure object.
    - bokeh_image -- bokeh.plotting.figure.image_rgba: Image object.
    - bokeh_img_ds -- bokeh...image_rgba.datasource: Image data.
    - async -- Bool; Set asynchronous.
    - unit_dict -- Dict; Dict of units for each variable.
    - stats_widget -- bokeh.models.widgets.Div; Stats widget object.
    - colorbar_widget -- bokeh.models.widgets.Div; C'bar widget object.
    - current_config -- Str; Current data configuration.
    - plot_description -- Str; Descriptive name of current config.
    - mslp_contour_label_dict -- Dict; Dict of MSLP levels and labels.
    - plot_funcs -- Dict; Dict of plotting methods.
    - update_funcs -- Dict; Dict of update methods.
    - coords_lat -- Numpy array; 1-D array of latitude coords.
    - coords_lon -- Numpy array; 1-D array of longitude coords.
    - quiver_plot -- matplotlib plot; Wind quiver plot.
    - mslp_contour -- matplotlib plot; MSLP contour plot.
    - wind_stream_plot -- matplotlib plot; Wind stream plot.
        
    """
    
    TITLE_TEXT_WIDTH = 40
    PRESSURE_LEVELS_HPA = range(980, 1030, 2)

    def __init__(self,
                 dataset,
                 po1,
                 figname,
                 plot_var,
                 conf1,
                 reg1,
                 rd1,
                 unit_dict,
                 app_path):
                 
        """Initialisation function for ForestPlot class."""
        
        self.region_dict = rd1
        self.main_plot = None
        self.current_time = 0
        self.plot_options = po1
        self.dataset = dataset
        self.figure_name = figname
        self.current_var = plot_var
        self._set_config_value(conf1)
        self.current_region = reg1
        self.app_path = app_path
        self.data_bounds = self.region_dict[self.current_region]
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
        self.stats_widget = None
        self.colorbar_widget = None
        
    def _set_config_value(self, new_config):
    
        """Set new config value.
        
        Arguments
        ---------
        
        - new_config - Str; Name of initial data configuration
        
        """
        
        self.current_config = new_config
        self.plot_description = self.dataset[
            self.current_config]['data_type_name']

    def setup_pressure_labels(self):
    
        """Create dict of pressure levels, for labelling contour plots.
        
        """
        
        self.mslp_contour_label_dict = {}
        for pressure1 in ForestPlot.PRESSURE_LEVELS_HPA:
            self.mslp_contour_label_dict[
                pressure1] = '{0:d}hPa'.format(int(pressure1))

    def setup_plot_funcs(self):
    
        """Set up dictionary of plot functions. 
        
        This is used by the main create_plot() function to call the 
        plotting function relevant to the specific variable being 
        plotted. There is also a second dictionary which is by the 
        update_plot() function, which does the minimum amount of work 
        to update the plot, and is used for some option changes,
        mainly a change in the forecast time selected.
        
        """
        
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
    
        """Update the latitude and longitude coordinates for the data.
        
        Arguments
        ---------
        
        - data_cube -- Cube; cube to take coordinates from.
        
        """

        self.coords_lat = data_cube.coords('latitude')[0].points
        self.coords_long = data_cube.coords('longitude')[0].points

    def create_blank(self):
    
        """Create blank plot."""
        
        self.main_plot = None
        self.current_title = 'Blank plot'

    def update_precip(self):
    
        """Update precipitation plots.
        
        Called by update_plot() when precipitation is the selected 
        plot type.
        
        """
        
        data_cube = self.dataset[self.current_config][
            'data'].get_data(self.current_var)
        array_for_update = data_cube[self.current_time].data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(data_cube)
        self.update_stats(data_cube)

    def plot_precip(self):
    
        """Create precipitation plots.
        
        Called by create_plot when precipitation is the selected plot 
        type.
        
        """
        
        data_cube = self.dataset[self.current_config][
            'data'].get_data(self.current_var)

        self.update_coords(data_cube)
        self.current_axes.coastlines(resolution='50m')
        self.main_plot = \
            self.current_axes.pcolormesh(self.coords_long,
                                         self.coords_lat,
                                         data_cube[
                                             self.current_time].data,
                                         cmap=self.plot_options[
                                             self.current_var]['cmap'],
                                         norm=self.plot_options[
                                             self.current_var]['norm'],
                                         edgecolor='face',
                                         transform=cartopy.crs.PlateCarree())
        self.update_title(data_cube)
        self.update_stats(data_cube)

    def update_wind_vectors(self):
    
        """Update wind vector plots.
        
        Called by update_plot() when wind vectors is the selected plot 
        type.
        
        """

        wind_speed_cube = self.dataset[self.current_config][
            'data'].get_data('wind_speed')
        array_for_update = wind_speed_cube[
            self.current_time].data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(wind_speed_cube)
        self.update_stats(wind_speed_cube)
        wv_u_data = self.dataset[self.current_config]['data'].get_data('wv_U')
        wv_v_data = self.dataset[self.current_config]['data'].get_data('wv_V')
        self.quiver_plot.set_UVC(wv_u_data[self.current_time],
                                 wv_v_data[self.current_time])

    def plot_wind_vectors(self):
    
        """Create wind vector plots.
        
        Called by create_plot when wind vectors is the selected plot
        type.
        
        """

        wind_speed_cube = self.dataset[self.current_config][
            'data'].get_data('wind_speed')
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

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical',
                                                            'coastline',
                                                            '50m',
                                                            edgecolor='0.5',
                                                            facecolor='none')
        self.current_axes.add_feature(coastline_50m)

        self.quiver_plot = \
            self.current_axes.quiver(
                self.dataset[self.current_config]['data'].get_data('wv_X'),
                self.dataset[self.current_config][
                    'data'].get_data('wv_Y'),
                self.dataset[self.current_config][
                    'data'].get_data('wv_U')[self.current_time],
                self.dataset[self.current_config][
                    'data'].get_data('wv_V')[self.current_time],
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
    
        """Update wind speed with MSLP contours plots. 
        
        Called by update_plot() when wind speed with MSLP is the 
        selected plot type.
        
        """
        
        wind_speed_cube = self.dataset[self.current_config][
            'data'].get_data('wind_speed')
        array_for_update = wind_speed_cube[
            self.current_time].data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        # to update contours, remove old elements and generate new contours
        for c1 in self.mslp_contour.collections:
            self.current_axes.collections.remove(c1)

        ap_cube = self.dataset[self.current_config]['data'].get_data('mslp')
        self.mslp_contour = \
            self.current_axes.contour(self.long_grid_mslp,
                                      self.lat_grid_mslp,
                                      ap_cube[
                                          self.current_time].data,
                                      levels=ForestPlot.PRESSURE_LEVELS_HPA,
                                      colors='k')
        self.current_axes.clabel(self.mslp_contour,
                                 inline=False,
                                 fmt=self.mslp_contour_label_dict)

        self.update_title(wind_speed_cube)
        self.update_stats(wind_speed_cube)

    def plot_wind_mslp(self):
    
        """Create wind speed with MSLP contour plots.
        
        Called by create_plot when wind speed with MSLP contours is the
        selected plot type.
        
        """
        
        wind_speed_cube = self.dataset[self.current_config][
            'data'].get_data('wind_speed')
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

        ap_cube = self.dataset[self.current_config]['data'].get_data('mslp')
        lat_mslp = ap_cube.coords('latitude')[0].points
        long_mslp = ap_cube.coords('longitude')[0].points
        self.long_grid_mslp, self.lat_grid_mslp = numpy.meshgrid(
            long_mslp, lat_mslp)
        self.mslp_contour = \
            self.current_axes.contour(self.long_grid_mslp,
                                      self.lat_grid_mslp,
                                      ap_cube[
                                          self.current_time].data,
                                      levels=ForestPlot.PRESSURE_LEVELS_HPA,
                                      colors='k')
        self.current_axes.clabel(self.mslp_contour,
                                 inline=False,
                                 fmt=self.mslp_contour_label_dict)

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical',
                                                            'coastline',
                                                            '50m',
                                                            edgecolor='0.5',
                                                            facecolor='none')

        self.current_axes.add_feature(coastline_50m)
        self.update_title(wind_speed_cube)

    def update_wind_streams(self):
    
        """Update wind streamline plots.
        
        Called by update_plot() when wind streamlines is the selected 
        plot type.
        
        """
        
        wind_speed_cube = self.dataset[self.current_config][
            'data'].get_data('wind_speed')
        array_for_update = wind_speed_cube[
            self.current_time].data[:-1, :-1].ravel()
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
                self.dataset[self.current_config]['data'].get_data(
                    'wv_X_grid'),
                self.dataset[self.current_config][
                    'data'].get_data('wv_Y_grid'),
                self.dataset[self.current_config][
                    'data'].get_data('wv_U')[self.current_time],
                self.dataset[self.current_config][
                    'data'].get_data('wv_V')[self.current_time],
                color='k',
                density=[0.5, 1.0])
        # we need to manually keep track of arrows so they can be removed when
        # the plot is updated
        pl2 = list(self.current_axes.patches)
        self.wind_stream_patches = [p1 for p1 in pl2 if p1 not in pl1]

    def plot_wind_streams(self):
    
        """Create wind streamline plots.
        
        Called by create_plot when wind streamlines is the selected 
        plot type.
        
        """
        
        wind_speed_cube = self.dataset[self.current_config][
            'data'].get_data('wind_speed')
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
                self.dataset[self.current_config]['data'].get_data(
                    'wv_X_grid'),
                self.dataset[self.current_config][
                    'data'].get_data('wv_Y_grid'),
                self.dataset[self.current_config][
                    'data'].get_data('wv_U')[self.current_time],
                self.dataset[self.current_config][
                    'data'].get_data('wv_V')[self.current_time],
                color='k',
                density=[0.5, 1.0])

        # we need to manually keep track of arrows so they can be removed when
        # the plot is updated
        pl2 = list(self.current_axes.patches)
        self.wind_stream_patches = [p1 for p1 in pl2 if p1 not in pl1]

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical',
                                                            'coastline',
                                                            '50m',
                                                            edgecolor='0.5',
                                                            facecolor='none')
        self.current_axes.add_feature(coastline_50m)
        self.update_title(wind_speed_cube)

    def update_air_temp(self):
    
        """Update air temperature plots.
        
        Called by update_plot() when air temperature is the selected
        plot type.
        
        """
        
        at_cube = self.dataset[self.current_config][
            'data'].get_data(self.current_var)
        array_for_update = at_cube[self.current_time].data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(at_cube)
        self.update_stats(at_cube)

    def plot_air_temp(self):
    
        """Create air temperature plots.
        
        Called by create_plot when air temperature is the selected plot
        type.
        
        """
        
        at_cube = self.dataset[self.current_config][
            'data'].get_data(self.current_var)
        self.update_coords(at_cube)
        self.main_plot = \
            self.current_axes.pcolormesh(self.coords_long,
                                         self.coords_lat,
                                         at_cube[
                                             self.current_time].data,
                                         cmap=self.plot_options[
                                             self.current_var]['cmap'],
                                         norm=self.plot_options[
                                             self.current_var]['norm']
                                         )

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical',
                                                            'coastline',
                                                            '50m',
                                                            edgecolor='0.5',
                                                            facecolor='none')
        self.current_axes.add_feature(coastline_50m)
        self.update_title(at_cube)
        self.update_stats(at_cube)

    def update_mslp(self):
    
        '''Update MSLP plots. 
    
        Called by update_plot() when MSLP is the selected plot type.
        
        '''
        
        ap_cube = self.dataset[self.current_config][
            'data'].get_data(self.current_var)
        array_for_update = ap_cube[self.current_time].data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(ap_cube)
        self.update_stats(ap_cube)

    def plot_mslp(self):
    
        """Create MSLP plots.
        
        Called by create_plot when MSLP is the selected plot type.
        
        """
        
        ap_cube = self.dataset[self.current_config][
            'data'].get_data(self.current_var)
        self.update_coords(ap_cube)
        self.main_plot = \
            self.current_axes.pcolormesh(self.coords_long,
                                         self.coords_lat,
                                         ap_cube[
                                             self.current_time].data,
                                         cmap=self.plot_options[
                                             self.current_var]['cmap'],
                                         norm=self.plot_options[
                                             self.current_var]['norm']
                                         )

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical',
                                                            'coastline',
                                                            '50m',
                                                            edgecolor='0.5',
                                                            facecolor='none')
        self.current_axes.add_feature(coastline_50m)
        self.update_title(ap_cube)
        self.update_stats(ap_cube)

    def update_cloud(self):
    
        """Update cloud fraction plots.
        
        Called by update_plot() when cloud fraction is the selected 
        plot type.
        
        """
        
        cloud_cube = self.dataset[self.current_config][
            'data'].get_data(self.current_var)
        array_for_update = cloud_cube[self.current_time].data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(cloud_cube)
        self.update_stats(cloud_cube)

    def plot_cloud(self):
    
        """Create cloud fraction plots.
        
        Called by create_plot when cloud fraction is the selected plot
        type.
        
        """
        
        cloud_cube = self.dataset[self.current_config][
            'data'].get_data(self.current_var)
        self.update_coords(cloud_cube)
        self.main_plot = \
            self.current_axes.pcolormesh(self.coords_long,
                                         self.coords_lat,
                                         cloud_cube[self.current_time].data,
                                         cmap=self.plot_options[
                                             self.current_var]['cmap'],
                                         norm=self.plot_options[
                                             self.current_var]['norm']
                                         )

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical',
                                                            'coastline',
                                                            '50m',
                                                            edgecolor='0.5',
                                                            facecolor='none')
        self.current_axes.add_feature(coastline_50m)
        self.update_title(cloud_cube)
        self.update_stats(cloud_cube)

    def update_him8(self):
    
        """Update himawari-8 image plots.
        
        Called by update_plot() when cloud fraction is the selected 
        plot type.
        
        """
        
        him8_image = self.dataset[
            'himawari-8']['data'].get_data(self.current_var)[self.current_time]
        self.current_axes.images.remove(self.main_plot)
        self.main_plot = self.current_axes.imshow(him8_image,
                                                  extent=(self.data_bounds[2],
                                                          self.data_bounds[3],
                                                          self.data_bounds[0],
                                                          self.data_bounds[1]),
                                                  origin='upper')
        self.update_title(None)

    def plot_him8(self):
    
        """Create himawari-8 image plots.
        
        Called by create_plot() when cloud fraction is the selected 
        plot type.
        
        """
        
        him8_image = self.dataset[
            'himawari-8']['data'].get_data(self.current_var)[self.current_time]
        self.main_plot = self.current_axes.imshow(him8_image,
                                                  extent=(self.data_bounds[2],
                                                          self.data_bounds[3],
                                                          self.data_bounds[0],
                                                          self.data_bounds[1]),
                                                  origin='upper')

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical',
                                                            'coastline',
                                                            '50m',
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
    
        """Update himawari-8 image plots. 
        
        Called by update_plot() when cloud fraction is the selected 
        plot type.
        
        """
        
        simim_cube = self.dataset['simim']['data'].get_data(
            self.current_var)[self.current_time]
        array_for_update = simim_cube.data[:-1, :-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(None)

    def plot_simim(self):
    
        """Create himawari-8 image plots.
        
        Called by create_plot() when cloud fraction is the selected 
        plot type.
        
        """

        simim_cube = self.dataset['simim']['data'].get_data(
            self.current_var)[self.current_time]
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
                                                            '50m',
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
    
        """Call functions to update simulated/himawari-8 image."""
        
        if self.current_config == 'himawari-8':
            self.update_him8()
        elif self.current_config == 'simim':
            self.update_simim()

    def plot_sat_simim_imagery(self):
    
        """Call functions to plot simulated/himawari-8 image."""
        
        if self.current_config == 'himawari-8':
            self.plot_him8()
        elif self.current_config == 'simim':
            self.plot_simim()

    def update_stats(self, current_cube):
    
        """Update plot stats widget."""
        
        stats_str_list = [self.current_title]
        unit_str = self.unit_dict[self.current_var]
        data_at_time = current_cube[self.current_time].data
        max_val = numpy.max(data_at_time)
        min_val = numpy.min(data_at_time)
        mean_val = numpy.mean(data_at_time)
        std_val = numpy.std(data_at_time)
        rms_val = numpy.sqrt(numpy.mean(numpy.power(data_at_time, 2.0)))

        stats_str_list += ['Max = {0:.4f} {1}'.format(max_val, unit_str)]
        stats_str_list += ['Min = {0:.4f} {1}'.format(min_val, unit_str)]
        stats_str_list += ['Mean = {0:.4f} {1}'.format(mean_val, unit_str)]
        stats_str_list += ['STD = {0:.4f} {1}'.format(std_val, unit_str)]
        stats_str_list += ['RMS = {0:.4f} {1}'.format(rms_val, unit_str)]

        self.stats_string = '</br>'.join(stats_str_list)

    def update_title(self, current_cube):

        """Update plot title.
        
        Arguments
        ---------
        
        - current_cube -- Cube; cube to take info for title from.
        
        """

        try:
            datestr1 = forest.util.get_time_str(
                current_cube.dim_coords[0].points[self.current_time])
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

        """Main plotting function. 
        
        Generic elements of the plot are created here, and then the 
        plotting function for the specific variable is called using 
        the self.plot_funcs dictionary.
        
        """

        self.create_matplotlib_fig()
        self.create_bokeh_fig()

        return self.bokeh_figure

    def create_matplotlib_fig(self):

        """Create matplotlib figure."""

        self.current_figure = matplotlib.pyplot.figure(self.figure_name,
                                                       figsize=(8.0, 6.0))
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

            self.current_figure.canvas.draw()

    def create_bokeh_fig(self):

        """Setup Bokeh figure."""

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
            bokeh.plotting.figure(plot_width=800,
                                  plot_height=600,
                                  x_range=x_limits,
                                  y_range=y_limits,
                                  tools='pan,wheel_zoom,reset,save')

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

        """Create Bokeh image data using matplotlib figure."""

        cur_region = self.region_dict[self.current_region]
        latitude_range = cur_region[1] - cur_region[0]
        longitude_range = cur_region[3] - cur_region[2]
        self.bokeh_image = \
            self.bokeh_figure.image_rgba(image=[self.current_img_array],
                                         x=[cur_region[2]],
                                         y=[cur_region[0]],
                                         dw=[longitude_range],
                                         dh=[latitude_range])
        self.bokeh_img_ds = self.bokeh_image.data_source

    def update_bokeh_img(self):

        """Update Bokeh image data using new matplotlib figure."""

        cur_region = self.region_dict[self.current_region]
        self.current_figure.set_figwidth(8)
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

        """Main plot update function. 
        
        Generic elements of the plot are updated here where possible, 
        and then the plot update function for the specific variable is 
        called using the self.plot_funcs dictionary.
        
        """

        self.update_funcs[self.current_var]()
        if self.use_mpl_title:
            self.current_axes.set_title(self.current_title)
        self.current_figure.canvas.draw_idle()
        if not self.async:
            self.update_bokeh_img()
        if self.stats_widget:
            self.update_stats_widget()

    def create_stats_widget(self):

        """Create Bokeh Div widget for stats."""

        self.stats_widget = bokeh.models.widgets.Div(text=self.stats_string,
                                                     height=200,
                                                     width=400,
                                                     )
        return self.stats_widget

    def create_colorbar_widget(self):

        """Create Bokeh Div widget for colorbar."""

        colorbar_html = "<img src='" + self.app_path + "/static/" + \
                        self.colorbar_link + "'\>"
            
        self.colorbar_widget = bokeh.models.widgets.Div(text=colorbar_html,
                                                        height=100,
                                                        width=800,
                                                        )
        return self.colorbar_widget

    def update_stats_widget(self):

        """Update stats based on change of var/time/config/region."""

        print('Updating stats widget')

        try:
            self.stats_widget.text = self.stats_string
        except AttributeError as e1:
            print('Unable to update stats as stats widget not initiated')

    def update_colorbar_widget(self):

        """Update colorbar based on change of variable."""

        self.colorbar_link = self.current_var + '_colorbar.png'
        colorbar_html = "<img src='" + self.app_path + "/static/" + \
                        self.colorbar_link + "'\>"

        try:
            self.colorbar_widget.text = colorbar_html
        except AttributeError as e1:
            print('Unable to update colorbar as colorbar widget not initiated')

    def set_data_time(self, new_time):

        """Set a new data time and update
        
        Arguments
        ---------
        
        - new_time -- Int; New data time to plot.
        
        """

        print('selected new time {0}'.format(new_time))

        self.current_time = new_time
        self.update_plot()

    def set_var(self, new_var):

        """Set a new variable and update.
        
        Arguments
        ---------
        
        - new_var -- Str; Name of new variable to plot.
        
        """

        print('selected new var {0}'.format(new_var))

        self.current_var = new_var
        self.create_matplotlib_fig()
        
        if not self.async:
            self.update_bokeh_img()
            if self.stats_widget:
                self.update_stats_widget()
            if self.colorbar_widget:
                self.update_colorbar_widget()

    def set_region(self, new_region):

        """Set a new region and update.
        
        Arguments
        ---------
        
        - new_region -- Str; Name of new region to plot.
        
        """

        print('selected new region {0}'.format(new_region))

        self.current_region = new_region
        self.data_bounds = self.region_dict[self.current_region]
        self.create_matplotlib_fig()
        
        if not self.async:
            self.update_bokeh_img()
            if self.stats_widget:
                self.update_stats_widget()

    def set_config(self, new_config):

        """Set a new config value and update.
        
        Arguments
        ---------
        
        - new_config -- Str; Name of new data configuration to plot.
        
        """

        print('setting new config {0}'.format(new_config))
        
        self._set_config_value(new_config)
        self.create_matplotlib_fig()
        
        if not self.async:
            self.update_bokeh_img()
            if self.stats_widget:
                self.update_stats_widget()

    def link_axes_to_other_plot(self, other_plot):

        """Set axes of Bokeh plots to be linked.
        
        Arguments
        ---------
        
        - other_plot -- ForestPlot object; Plot to link axes with.
        
        """

        try:
            self.bokeh_figure.x_range = other_plot.bokeh_figure.x_range
            self.bokeh_figure.y_range = other_plot.bokeh_figure.y_range
        except:
            print('bokeh plot linking failed.')