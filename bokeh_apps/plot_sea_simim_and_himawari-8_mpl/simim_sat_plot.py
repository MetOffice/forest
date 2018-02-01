import textwrap

import matplotlib.pyplot
import cartopy
import cartopy.crs

import bokeh.models
import bokeh.plotting

import forest.util

class SimimSatPlot(object):
    '''
    Main plot class. The plotting function is create_plot().
    '''
    TITLE_TEXT_WIDTH = 40

    def __init__(self, datasets, po1, figname, plot_type, time, conf):
        '''
        Initialisation function for SEA_plot class
        '''
        self.main_plot = None
        self.plot_options = po1
        self.datasets = datasets
        self.figure_name = figname
        self.current_type = plot_type
        self.current_time = time
        self.set_config(conf)
        self.use_mpl_title = False
        self.show_axis_ticks = False
        self.show_colorbar = False
        self.setup_plot_funcs()

    def setup_plot_funcs(self):

        '''
        Set up dictionary of plot functions. This is used by the main
        create_plot() function to call the plotting function relevant to the
        specific variable being plotted. There is also a second dictionary
        which is by the update_plot() function, which does the minimum amount
        of work to update the plot, and is used for some option changes, mainly
        a change in the forecast time selected.
        '''

        self.plot_funcs = {'simim': self.plot_simim,
                           'himawari-8': self.plot_him8,
                           }
        self.update_funcs = {'simim': self.update_simim,
                             'himawari-8': self.update_him8,
                             }

    def set_config(self, new_config):
        self.current_config = new_config
        self.plot_description = self.datasets[self.current_config]['data_type_name']

    def update_him8(self):

        '''
        Update function for himawari-8 image plots, called by update_plot()
        when cloud fraction is the selected plot type.
        '''

        '''
        cloud_cube = datasets['himawari-8'][self.current_type]
        array_for_update = cloud_cube[self.current_time].data[:-1,:-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title()'''

    def plot_him8(self):

        '''
        Function for creating himawari-8 image plots, called by create_plot()
        when cloud fraction is the selected plot type.
        '''

        him8_image = self.datasets['himawari-8']['data'][self.current_type][self.current_time]
        self.current_axes.imshow(him8_image, extent=(90, 154, -18, 30),
                                 origin='upper')

        # Add coastlines to the map created by contourf.
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical',
                                                            'coastline',
                                                            '50m',
                                                            edgecolor='g',
                                                            alpha=0.5,
                                                            facecolor='none')

        self.current_axes.add_feature(coastline_50m)
        self.current_axes.set_extent((90, 154, -18, 30))

        self.update_title()

    def update_simim(self):

        '''
        Update function for himawari-8 image plots, called by update_plot() when
        cloud fraction is the selected plot type.
        '''

        '''simim_cube = datasets['simim']['data'][self.current_type][self.current_time]
        array_for_update = cloud_cube.data[:-1,:-1].ravel()
        self.main_plot.set_array(array_for_update)
        self.update_title()'''

    def plot_simim(self):

        '''
        Function for creating himawari-8 image plots, called by create_plot()
        when cloud fraction is the selected plot type.
        '''

        simim_cube = self.datasets['simim']['data'][self.current_type][self.current_time]
        lats = simim_cube.coords('grid_latitude')[0].points
        lons = simim_cube.coords('grid_longitude')[0].points
        self.main_plot = self.current_axes.pcolormesh(lons,
                                                      lats,
                                                      simim_cube.data,
                                                      cmap=self.plot_options[self.current_type]['cmap'],
                                                      norm=self.plot_options[self.current_type]['norm']
                                                      )

        # Add coastlines to the map created by contourf
        coastline_50m = cartopy.feature.NaturalEarthFeature('physical',
                                                            'coastline',
                                                            '50m',
                                                            edgecolor='g',
                                                            alpha=0.5,
                                                            facecolor='none')

        self.current_axes.add_feature(coastline_50m)
        self.current_axes.set_extent((90, 154, -18, 30))
        self.update_title()

    def update_title(self):

        '''
        Update plot title.
        '''

        str1 = '{plot_desc} {var_name} at {time}'.format(plot_desc=self.plot_description,
                                                         var_name=self.current_type,
                                                         time=self.current_time,
                                                         )
        self.current_title = '\n'.join(textwrap.wrap(str1,
                                                     SimimSatPlot.TITLE_TEXT_WIDTH))
        self.current_title = str1

    def create_plot(self):

        '''
        Main plotting function. Generic elements of the plot are created here, and then the plotting
        function for the specific variable is called using the self.plot_funcs dictionary.
        '''

        self.create_matplotlib_fig()
        self.create_bokeh_img_plot_from_fig()

        return self.bokeh_figure

    def update_plot(self):

        '''
        Main plot update function. Generic elements of the plot are updated here where possible, and then
        the plot update function for the specific variable is called using the self.plot_funcs dictionary.
        '''

        self.update_funcs[self.current_type]()
        if self.use_mpl_title:
            self.current_axes.set_title(self.current_title)
        self.current_figure.canvas.draw_idle()

        self.update_bokeh_img_plot_from_fig()

    def create_matplotlib_fig(self):

        '''
        A method for creating a matplotlib data plot.
        '''

        self.current_figure = matplotlib.pyplot.figure(self.figure_name, figsize=(8.0, 6.0))
        self.current_figure.clf()
        self.current_axes = self.current_figure.add_subplot(111, projection=cartopy.crs.PlateCarree())
        self.current_axes.set_position([0, 0, 1, 1])

        self.plot_funcs[self.current_config]()
        if self.use_mpl_title:
            self.current_axes.set_title(self.current_title)
        self.current_axes.set_xlim(90, 154)
        self.current_axes.set_ylim(-18, 30)
        self.current_axes.xaxis.set_visible(self.show_axis_ticks)
        self.current_axes.yaxis.set_visible(self.show_axis_ticks)
        if self.show_colorbar:
            self.current_figure.colorbar(self.main_plot,
                                         orientation='horizontal')

        self.current_figure.canvas.draw()

    def create_bokeh_img_plot_from_fig(self):

        '''
        A method to create a Bokeh figure displaying a matplotlib-generated
         image.
        '''

        self.current_img_array = forest.util.get_image_array_from_figure(self.current_figure)
        print('size of image array is {0}'.format(self.current_img_array.shape))

        # Set figure navigation limits
        x_limits = bokeh.models.Range1d(90, 154, bounds=(90, 154))
        y_limits = bokeh.models.Range1d(-18, 30, bounds=(-18, 30))

        # Initialize figure
        self.bokeh_figure = bokeh.plotting.figure(plot_width=800,
                                                  plot_height=600,
                                                  x_range=x_limits,
                                                  y_range=y_limits,
                                                  tools='pan,wheel_zoom,reset,save')

        latitude_range = 48
        longitude_range = 64
        self.bokeh_image = self.bokeh_figure.image_rgba(image=[self.current_img_array],
                                                        x=[90],
                                                        y=[-18],
                                                        dw=[longitude_range],
                                                        dh=[latitude_range])
        self.bokeh_figure.title.text = self.current_title

        self.bokeh_img_ds = self.bokeh_image.data_source

    def update_bokeh_img_plot_from_fig(self):

        '''

        '''
        self.current_img_array = forest.util.get_image_array_from_figure(self.current_figure)
        self.bokeh_figure.title.text = self.current_title
        self.bokeh_img_ds.data[u'image'] = [self.current_img_array]

    def share_axes(self, axes_list):

        '''
        Sets the axes of this plot to be shared with the other axes in the supplied list.
        '''

        self.current_axes.get_shared_x_axes().join(self.current_axes, *axes_list)
        self.current_axes.get_shared_y_axes().join(self.current_axes, *axes_list)

    def set_data_time(self, new_data_time):
        self.current_time = new_data_time
        self.create_matplotlib_fig()
        self.update_bokeh_img_plot_from_fig()

    def set_type(self, new_type):
        self.current_type = new_type
        self.create_matplotlib_fig()
        self.update_bokeh_img_plot_from_fig()



    def link_axes_to_other_plot(self, other_plot):
        try:
            self.bokeh_figure.x_range = other_plot.bokeh_figure.x_range
            self.bokeh_figure.y_range = other_plot.bokeh_figure.y_range
        except:
            print('bokeh plot linking failed.')