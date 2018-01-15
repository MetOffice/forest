import textwrap

import numpy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot
import matplotlib.cm

import cartopy
import cartopy.crs
import cartopy.io.img_tiles

import lib_sea


import bokeh.io
import bokeh.layouts 
import bokeh.models.widgets
import bokeh.plotting

class SEA_plot(object):
    '''
    Main plot class. The plotting function is create_plot().
    '''
    TITLE_TEXT_WIDTH = 40
    PRESSURE_LEVELS_HPA = range(980,1030,2)
    def __init__(self, dataset, po1,figname, plot_var, conf1, reg1, rd1):
        '''
        Initialisation function for SEA_plot class
        '''
        self.region_dict = rd1
        self.main_plot = None
        self.current_time = 0
        self.plot_options = po1
        self.dataset = dataset
        self.figure_name = figname
        self.current_var = plot_var
        self.current_config = conf1
        self.current_region = reg1
        self.data_bounds = self.region_dict[self.current_region]
        self.plot_description = self.dataset[self.current_config]['model_name']
        self.show_colorbar = False
        self.show_axis_ticks = False
        self.use_mpl_title = False
        self.setup_plot_funcs()
        self.setup_pressure_labels()
    
    def setup_pressure_labels(self):
        '''
        Create dict of pressure levels, to be used labelling MSLP contour plots.
        '''
        self.mslp_contour_label_dict = {}
        for pressure1 in SEA_plot.PRESSURE_LEVELS_HPA:
            self.mslp_contour_label_dict[pressure1] = '{0:d}hPa'.format(int(pressure1))    

    
    def setup_plot_funcs(self):
        '''
        Set up dictionary of plot functions. This is used by the main create_plot() function 
        to call the plotting function relevant to the specific variable being plotted. There
        is also a second dictionary which is by the update_plot() function, which does the minimum
        amount of work to update the plot, and is used for some option changes, mainly a change 
        in the forecast time selected.
        '''
        self.plot_funcs = {'precipitation' : self.plot_precip,
                           'wind_vectors' : self.plot_wind_vectors,
                           'wind_mslp' : self.plot_wind_mslp,
                           'wind_streams' : self.plot_wind_streams,
                           'mslp' : self.plot_mslp,
                           'air_temperature' : self.plot_air_temp,
                           'cloud_fraction' :self.plot_cloud,
                          }
        self.update_funcs =  {'precipitation' : self.update_precip,
                           'wind_vectors' : self.update_wind_vectors,
                           'wind_mslp' : self.update_wind_mslp,
                              'wind_streams' : self.update_wind_streams,
                           'mslp' : self.update_mslp,
                           'air_temperature' : self.update_air_temp,
                           'cloud_fraction' :self.update_cloud,
                          }
        
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
        data_cube = self.dataset[self.current_config][self.current_var]
        array_for_update = data_cube[self.current_time].data[:-1,:-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(data_cube)
        
    def plot_precip(self):
        '''
        Function for creating precipitation plots, called by create_plot when 
        precipitation is the selected plot type.
        '''
        data_cube = self.dataset[self.current_config][self.current_var]

        self.update_coords(data_cube)
        self.current_axes.coastlines(resolution='110m')
        self.main_plot = self.current_axes.pcolormesh(self.coords_long, 
                                     self.coords_lat, 
                                     data_cube[self.current_time].data, 
                                     cmap=self.plot_options[self.current_var]['cmap'],
                                     norm=self.plot_options[self.current_var]['norm'],
                                     transform=cartopy.crs.PlateCarree())
        self.update_title(data_cube)

    def update_wind_vectors(self):
        '''
        Update function for wind vector plots, called by update_plot() when 
        wind vectors is the selected plot type.
        '''
        
        wind_speed_cube = self.dataset[self.current_config]['wind_speed']
        array_for_update = wind_speed_cube[self.current_time].data[:-1,:-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(wind_speed_cube)
       
        self.quiver_plot.set_UVC(self.dataset[self.current_config]['wv_U'][self.current_time],
                                 self.dataset[self.current_config]['wv_V'][self.current_time],
                                )
    
    def plot_wind_vectors(self):
        '''
        Function for creating wind vector plots, called by create_plot when 
        wind vectors is the selected plot type.
        '''
        wind_speed_cube = self.dataset[self.current_config]['wind_speed']
        self.update_coords(wind_speed_cube)
        self.main_plot = self.current_axes.pcolormesh(self.coords_long, 
                                                      self.coords_lat, 
                                                      wind_speed_cube[self.current_time].data,
                                                      cmap=self.plot_options[self.current_var]['cmap'],
                                                      norm=self.plot_options[self.current_var]['norm']
                                                     )

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical', 
                                                            'coastline', 
                                                            '50m', 
                                                            edgecolor='0.5', 
                                                            facecolor = 'none')
        self.current_axes.add_feature(coastline_50m)
        
        #print(data_wv['xs'])
        self.quiver_plot = self.current_axes.quiver(self.dataset[self.current_config]['wv_X'],
                                               self.dataset[self.current_config]['wv_Y'],
                                               self.dataset[self.current_config]['wv_U'][self.current_time],
                                               self.dataset[self.current_config]['wv_V'][self.current_time],
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
        '''
        Update function for wind speed with MSLP contours plots, called by update_plot() when 
        wind speed with MSLP is the selected plot type.
        '''
        wind_speed_cube = self.dataset[self.current_config]['wind_speed']
        array_for_update = wind_speed_cube[self.current_time].data[:-1,:-1].ravel()
        self.main_plot.set_array(array_for_update)
        # to update contours, remove old elements and generate new contours
        for c1 in self.mslp_contour.collections:
            self.current_axes.collections.remove(c1)
            
        ap_cube = self.dataset[self.current_config]['mslp']            
        self.mslp_contour = self.current_axes.contour(self.long_grid_mslp,
                                                      self.lat_grid_mslp,
                                                      ap_cube[self.current_time].data,
                                                      levels=SEA_plot.PRESSURE_LEVELS_HPA,
                                                      colors='k')
        self.current_axes.clabel(self.mslp_contour, 
                                 inline=False, 
                                 fmt=self.mslp_contour_label_dict)        
        
        self.update_title(wind_speed_cube)

        
    def plot_wind_mslp(self):
        '''
        Function for creating wind speed with MSLP contour plots, called by create_plot when 
        wind speed with MSLP contours is the selected plot type.
        '''
        wind_speed_cube = self.dataset[self.current_config]['wind_speed']
        self.update_coords(wind_speed_cube)
        self.main_plot = self.current_axes.pcolormesh(self.coords_long, 
                                                      self.coords_lat, 
                                                      wind_speed_cube[self.current_time].data,
                                                      cmap=self.plot_options[self.current_var]['cmap'],
                                                      norm=self.plot_options[self.current_var]['norm']
                                                     )
        
        ap_cube = self.dataset[self.current_config]['mslp']
        lat_mslp = ap_cube.coords('latitude')[0].points
        long_mslp = ap_cube.coords('longitude')[0].points
        self.long_grid_mslp, self.lat_grid_mslp = numpy.meshgrid(long_mslp, lat_mslp)
        self.mslp_contour = self.current_axes.contour(self.long_grid_mslp,
                                                      self.lat_grid_mslp,
                                                      ap_cube[self.current_time].data,
                                                      levels=SEA_plot.PRESSURE_LEVELS_HPA,
                                                      colors='k')
        self.current_axes.clabel(self.mslp_contour, 
                                 inline=False, 
                                 fmt=self.mslp_contour_label_dict)


        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical', 
                                                            'coastline', 
                                                            '50m', 
                                                            edgecolor='0.5', 
                                                            facecolor = 'none')
        
        self.current_axes.add_feature(coastline_50m)    
        self.update_title(wind_speed_cube)

    def update_wind_streams(self):
        '''
        Update function for wind streamline plots, called by update_plot() when 
        wind streamlines is the selected plot type.
        '''
        wind_speed_cube = self.dataset[self.current_config]['wind_speed']
        array_for_update = wind_speed_cube[self.current_time].data[:-1,:-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(wind_speed_cube)
        
        # remove old plot elements if they are still present
        self.current_axes.collections.remove(self.wind_stream_plot.lines)
        for p1 in self.wind_stream_patches:
            self.current_axes.patches.remove(p1)
        
        pl1 = list(self.current_axes.patches)    
        self.wind_stream_plot = self.current_axes.streamplot(self.dataset[self.current_config]['wv_X_grid'],
                                      self.dataset[self.current_config]['wv_Y_grid'],
                                      self.dataset[self.current_config]['wv_U'][self.current_time],
                                      self.dataset[self.current_config]['wv_V'][self.current_time],
                                      color='k',
                                      density=[0.5,1.0])   
        # we need to manually keep track of arrows so they can be removed when the plot is updated
        pl2 = list(self.current_axes.patches)
        self.wind_stream_patches = [p1 for p1 in pl2 if p1 not in pl1]
        
        
       
    def plot_wind_streams(self):
        '''
        Function for creating wind streamline plots, called by create_plot when 
        wind streamlines is the selected plot type.
        '''
        wind_speed_cube = self.dataset[self.current_config]['wind_speed']
        self.update_coords(wind_speed_cube)
        self.main_plot = self.current_axes.pcolormesh(self.coords_long, 
                                                      self.coords_lat, 
                                                      wind_speed_cube[self.current_time].data,
                                                      cmap=self.plot_options[self.current_var]['cmap'],
                                                      norm=self.plot_options[self.current_var]['norm']
                                                     )
        pl1 = list(self.current_axes.patches)
        self.wind_stream_plot = self.current_axes.streamplot(self.dataset[self.current_config]['wv_X_grid'],
                                      self.dataset[self.current_config]['wv_Y_grid'],
                                      self.dataset[self.current_config]['wv_U'][self.current_time],
                                      self.dataset[self.current_config]['wv_V'][self.current_time],
                                      color='k',
                                      density=[0.5,1.0])
        
        
        # we need to manually keep track of arrows so they can be removed when the plot is updated
        pl2 = list(self.current_axes.patches)
        self.wind_stream_patches = [p1 for p1 in pl2 if p1 not in pl1]

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical', 
                                                            'coastline', 
                                                            '50m', 
                                                            edgecolor='0.5', 
                                                            facecolor = 'none')
        self.current_axes.add_feature(coastline_50m)    
        self.update_title(wind_speed_cube)        

    def update_air_temp(self):
        '''
        Update function for air temperature plots, called by update_plot() when 
        air temperature is the selected plot type.
        '''
        at_cube = self.dataset[self.current_config][self.current_var]
        array_for_update = at_cube[self.current_time].data[:-1,:-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(at_cube)
        
    def plot_air_temp(self):
        '''
        Function for creating air temperature plots, called by create_plot when 
        air temperature is the selected plot type.
        '''
        at_cube = self.dataset[self.current_config][self.current_var]
        self.update_coords(at_cube)
        self.main_plot = self.current_axes.pcolormesh(self.coords_long, 
                                                      self.coords_lat, 
                                                      at_cube[self.current_time].data,
                                                      cmap=self.plot_options[self.current_var]['cmap'],
                                                      norm=self.plot_options[self.current_var]['norm']
                                                     )

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical', 
                                                            'coastline', 
                                                            '50m', 
                                                            edgecolor='0.5', 
                                                            facecolor = 'none')
        self.current_axes.add_feature(coastline_50m)    
        self.update_title(at_cube)

    def update_mslp(self):
        '''
        Update function for MSLP plots, called by update_plot() when 
        MSLP is the selected plot type.
        '''
        ap_cube = self.dataset[self.current_config][self.current_var]
        array_for_update = ap_cube[self.current_time].data[:-1,:-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(ap_cube)
    
    def plot_mslp(self):
        '''
        Function for creating MSLP plots, called by create_plot when 
        MSLP is the selected plot type.
        '''
        ap_cube = self.dataset[self.current_config][self.current_var]
        self.update_coords(ap_cube)
        self.main_plot = self.current_axes.pcolormesh(self.coords_long, 
                                                      self.coords_lat, 
                                                      ap_cube[self.current_time].data,
                                                      cmap=self.plot_options[self.current_var]['cmap'],
                                                      norm=self.plot_options[self.current_var]['norm']
                                                     )

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical', 
                                                            'coastline', 
                                                            '50m', 
                                                            edgecolor='0.5', 
                                                            facecolor = 'none')
        self.current_axes.add_feature(coastline_50m)    
        self.update_title(ap_cube)
        

    def update_cloud(self):
        '''
        Update function for cloud fraction plots, called by update_plot() when 
        cloud fraction is the selected plot type.
        '''
        cloud_cube = self.dataset[self.current_config][self.current_var]
        array_for_update = cloud_cube[self.current_time].data[:-1,:-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title(cloud_cube)
    
    def plot_cloud(self):
        '''
        Function for creating cloud fraction plots, called by create_plot when 
        cloud fraction is the selected plot type.
        '''
        cloud_cube = self.dataset[self.current_config][self.current_var]
        self.update_coords(cloud_cube)
        self.main_plot = self.current_axes.pcolormesh(self.coords_long, 
                                                      self.coords_lat, 
                                                      cloud_cube[self.current_time].data,
                                                      cmap=self.plot_options[self.current_var]['cmap'],
                                                      norm=self.plot_options[self.current_var]['norm']
                                                     )

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical', 
                                                            'coastline', 
                                                            '50m', 
                                                            edgecolor='0.5', 
                                                            facecolor = 'none')
        self.current_axes.add_feature(coastline_50m)   
        self.update_title(cloud_cube)
 
    def update_title(self, current_cube):
        '''
        Update plot title.
        '''
        datestr1 = lib_sea.get_time_str(current_cube.dim_coords[0].points[self.current_time])
        
        str1 = '{plot_desc} {var_name} at {fcst_time}'.format(var_name=self.current_var,
                                                              fcst_time=datestr1,
                                                              plot_desc=self.plot_description,
                                                             )
        self.current_title = '\n'.join(textwrap.wrap(str1, 
                                                     SEA_plot.TITLE_TEXT_WIDTH)) 
        
    def create_plot(self):
        '''
        Main plotting function. Generic elements of the plot are created here, and then the plotting
        function for the specific variable is called using the self.plot_funcs dictionary.
        '''
        self.create_matplotlib_fig()
        self.create_bokeh_img_plot_from_fig()
        
        return self.bokeh_figure
    
    def create_matplotlib_fig(self):
        self.current_figure = matplotlib.pyplot.figure(self.figure_name,
                                                       figsize=(4.0,3.0))
        self.current_figure.clf()
        self.current_axes = self.current_figure.add_subplot(111, projection=cartopy.crs.PlateCarree())
        self.current_axes.set_position([0, 0, 1, 1])

        self.plot_funcs[self.current_var]()
        if self.use_mpl_title:
            self.current_axes.set_title(self.current_title)
        self.current_axes.set_xlim(self.data_bounds[2], self.data_bounds[3])
        self.current_axes.set_ylim(self.data_bounds[0], self.data_bounds[1])
        self.current_axes.xaxis.set_visible(self.show_axis_ticks)
        self.current_axes.yaxis.set_visible(self.show_axis_ticks)
        if self.show_colorbar:
            self.current_figure.colorbar(self.main_plot,
                                         orientation='horizontal')

        #self.current_figure.tight_layout()
        self.current_figure.canvas.draw()
        
    def create_bokeh_img_plot_from_fig(self):
        self.current_img_array = lib_sea.get_image_array_from_figure(self.current_figure)
        print('size of image array is {0}'.format(self.current_img_array.shape))
        
        cur_region = self.region_dict[self.current_region]
        self.bokeh_figure = bokeh.plotting.figure(plot_width=800, 
                                                  plot_height=600, 
                                                  x_range=(cur_region[2], cur_region[3]),
                                                  y_range=(cur_region[0], cur_region[1]), 
                                                  toolbar_location=None,
                                                  tools = 'pan,wheel_zoom,reset')
        latitude_range = cur_region[1] - cur_region[0]
        longitude_range = cur_region[3] - cur_region[2]
        self.bokeh_image = self.bokeh_figure.image_rgba(image=[self.current_img_array], 
                                                   x=[cur_region[2]], 
                                                   y=[cur_region[0]], 
                                                   dw=[longitude_range], 
                                                   dh=[latitude_range])
        self.bokeh_figure.title.text = self.current_title
        
        self.bokeh_img_ds = self.bokeh_image.data_source
        
    def update_bokeh_img_plot_from_fig(self):
        self.current_img_array = lib_sea.get_image_array_from_figure(self.current_figure)
        self.bokeh_img_ds.data[u'image'] = [self.current_img_array]

    def update_plot(self):
        '''
        Main plot update function. Generic elements of the plot are updated here where possible, and then
        the plot update function for the specific variable is called using the self.plot_funcs dictionary.
        '''
        self.update_funcs[self.current_var]()
        self.current_axes.set_title(self.current_title)
        self.current_figure.canvas.draw_idle()
        
        self.update_bokeh_img_plot_from_fig()
        
        
    def on_data_time_change(self, attr1, old_val, new_val):
        '''
        Event handler for a change in the selected forecast data time.
        '''
        print('selected new time {0}'.format(new_val))
        self.current_time = new_val
        self.update_plot()
        
    def on_var_change(self, attr1, old_val, new_val):
        '''
        Event handler for a change in the selected plot type.
        '''
        print('selected new var {0}'.format(new_val))
        self.current_var = new_val
        self.create_matplotlib_fig()
        self.update_bokeh_img_plot_from_fig()
        
    def on_region_change(self, attr1, old_val, new_val):
        '''
        Event handler for a change in the selected plot region.
        '''
        print('selected new region {0}'.format(new_val))
        
        self.current_region = new_val
        self.data_bounds = self.region_dict[self.current_region]
        self.create_matplotlib_fig()
        self.update_bokeh_img_plot_from_fig()
    
    def on_config_change(self, attr1, old_val, new_val):
        '''
        Event handler for a change in the selected model configuration output.
        '''
        print('selected new config {0}'.format(new_val))
        self.current_config = new_val       
        self.create_matplotlib_fig()
        self.update_bokeh_img_plot_from_fig()

