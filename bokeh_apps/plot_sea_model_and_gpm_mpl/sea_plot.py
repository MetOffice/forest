import textwrap

import matplotlib
import matplotlib.pyplot
import cartopy.crs

import bokeh
import bokeh.models
import bokeh.plotting

import forest.util

class SEA_plot(object):
    '''
    Main plot class. The plotting function is create_plot().
    '''
    TITLE_TEXT_WIDTH = 40

    def __init__(self,
                 dataset,
                 po1,
                 figname,
                 plot_type,
                 conf1,
                 reg1,
                 reg_dict1):
        '''
        Initialisation function for SEA_plot class
        '''
        self.main_plot = None
        self.current_time = 0
        self.plot_options = po1
        self.dataset = dataset
        self.figure_name = figname
        self.current_type = plot_type
        self._set_config_value(conf1)
        self.current_region = reg1
        self.region_dict = reg_dict1
        self.data_bounds = self.region_dict[self.current_region]
        self.use_mpl_title = False
        self.show_axis_ticks = False
        self.show_colorbar = False
        self.setup_plot_funcs()
        self.current_title = ''

    def setup_plot_funcs(self):

        '''
        Set up dictionary of plot functions. This is used by the main create_plot() function
        to call the plotting function relevant to the specific variable being plotted. There
        is also a second dictionary which is by the update_plot() function, which does the minimum
        amount of work to update the plot, and is used for some option changes, mainly a change
        in the forecast time selected.
        '''
        self.plot_funcs = {'precipitation': self.plot_precip}

    def _set_config_value(self, new_config):
        self.current_config = new_config
        self.plot_description = self.dataset[self.current_config]['data_type_name']


    def update_coords(self, data_cube):

        '''
        Update the latitude and longitude coordinates for the data.
        '''

        self.coords_lat = data_cube.coords('latitude')[0].points
        self.coords_long = data_cube.coords('longitude')[0].points

    def plot_precip(self):

        '''
        Function for creating precipitation plots, called by create_plot when
        precipitation is the selected plot type.
        '''

        data_cube = self.dataset[self.current_config][self.current_type]

        self.update_coords(data_cube)
        self.current_axes.coastlines(resolution='110m')
        self.main_plot = self.current_axes.pcolormesh(self.coords_long,
                                                      self.coords_lat,
                                                      data_cube[self.current_time].data,
                                                      cmap=self.plot_options[self.current_type]['cmap'],
                                                      norm=self.plot_options[self.current_type]['norm'],
                                                      transform=cartopy.crs.PlateCarree())
        self.update_title(data_cube)

    def update_title(self, current_cube):
        '''
        Update plot title.
        '''
        datestr1 = forest.util.get_time_str(current_cube.dim_coords[0].points[self.current_time])

        str1 = '{plot_desc} {var_name} at {fcst_time}'.format(var_name=self.current_type,
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

        '''
        A method for creating a matplotlib data plot.
        '''

        self.current_figure = matplotlib.pyplot.figure(self.figure_name, figsize=(4.0, 3.0))
        self.current_figure.clf()
        self.current_axes = self.current_figure.add_subplot(111, projection=cartopy.crs.PlateCarree())
        self.current_axes.set_position([0, 0, 1, 1])

        self.plot_funcs[self.current_type]()
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

    def set_data_time(self, new_time):
        self.current_time = new_time
        self.create_matplotlib_fig()
        self.update_bokeh_img_plot_from_fig()

    def set_type(self, new_type):
        self.current_type = new_type
        self.create_matplotlib_fig()
        self.update_bokeh_img_plot_from_fig()

    def set_config(self, new_config):
        self._set_config_value(new_config)
        self.create_matplotlib_fig()
        self.update_bokeh_img_plot_from_fig()

    def link_axes_to_other_plot(self, other_plot):
        try:
            self.bokeh_figure.x_range = other_plot.bokeh_figure.x_range
            self.bokeh_figure.y_range = other_plot.bokeh_figure.y_range
        except:
            print('bokeh plot linking failed.')