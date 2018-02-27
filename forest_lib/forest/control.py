import functools
import threading

import bokeh.models.widgets
import bokeh.layouts
import bokeh.plotting

import tornado

import forest.data


# set up bokeh widgets
def create_dropdown_opt_list(iterable1):
    return [(k1, k1) for k1 in iterable1]


class ForestController(object):

    """

    """

    def __init__(self,
                 times1,
                 init_time_ix,
                 datasets,
                 plot_names,
                 plots,
                 bokeh_imgs,
                 colorbar_widgets,
                 stats_widgets,
                 region_dict,
                 bokeh_doc):

        """

        """

        self.plots = plots
        self.bokeh_imgs = bokeh_imgs
        self.region_dict = region_dict
        self.plot_names = plot_names
        self.datasets = datasets
        self.available_times = times1
        self.init_time_index = init_time_ix
        self.init_time = times1[init_time_ix]
        self.num_times = self.available_times.shape[0]
        self.colorbar_widgets = colorbar_widgets
        self.stats_widgets = stats_widgets
        self.create_widgets()
        self.bokeh_doc = bokeh_doc


    def create_widgets(self):

        """

        """

        self.model_var_list_desc = 'Attribute to visualise'

        self.model_var_dd = \
            bokeh.models.widgets.Dropdown(label=self.model_var_list_desc,
                                          menu=create_dropdown_opt_list(
                                              self.plot_names),
                                          button_type='warning')
        self.model_var_dd.on_change('value', self.on_var_change)

        self.data_time_slider = \
            bokeh.models.widgets.Slider(start=0,
                                        end=self.num_times,
                                        value=self.init_time_index,
                                        step=1,
                                        title="Data time")

        self.data_time_slider.on_change('value', self.on_data_time_change)

        self.region_desc = 'Region'

        region_menu_list = create_dropdown_opt_list(self.region_dict.keys())
        self.region_dd = bokeh.models.widgets.Dropdown(menu=region_menu_list,
                                                       label=self.region_desc,
                                                       button_type='warning')
        self.region_dd.on_change('value', self.on_region_change)

        dataset_menu_list = create_dropdown_opt_list(self.datasets.keys())
        left_model_desc = 'Left display'

        self.left_model_dd = \
            bokeh.models.widgets.Dropdown(menu=dataset_menu_list,
                                          label=left_model_desc,
                                          button_type='warning')
        self.left_model_dd.on_change('value',
                                     functools.partial(self.on_config_change,
                                                       0))

        right_model_desc = 'Right display'
        self.right_model_dd = \
            bokeh.models.widgets.Dropdown(menu=dataset_menu_list,
                                          label=right_model_desc,
                                          button_type='warning')
        self.right_model_dd.on_change('value',
                                      functools.partial(self.on_config_change,
                                                        1))

        # layout widgets
        self.param_row = bokeh.layouts.row(self.model_var_dd, self.region_dd)
        self.slider_row = bokeh.layouts.row(self.data_time_slider)
        self.config_row = bokeh.layouts.row(
            self.left_model_dd, self.right_model_dd)
        self.plots_row = bokeh.layouts.row(*self.bokeh_imgs)
        self.colorbar_row = bokeh.layouts.row(*self.colorbar_widgets)
        self.stats_row = bokeh.layouts.row(*self.stats_widgets)

        self.main_layout = bokeh.layouts.column(self.param_row,
                                                self.slider_row,
                                                self.config_row,
                                                self.plots_row,
                                                self.colorbar_row,
                                                self.stats_row,
                                                )

    def on_data_time_change(self, attr1, old_val, new_val):

        '''Event handler for a change in the selected forecast data time.

        '''

        print('data time handler')

        for p1 in self.plots:
            new_time1 = self.available_times[new_val]
            p1.set_data_time(new_time1)


    def _refresh_times(self, new_var):
        self.available_times = forest.data.get_available_times(self.datasets,
                                                          new_var)
        try:
            new_time = self.available_times[self.data_time_slider.value]
        except IndexError:
            new_time = self.available_times[0]
            self.data_time_slider.value = 0

        self.data_time_slider.end = self.available_times.shape[0]
        return new_time


    def on_var_change(self, attr1, old_val, new_val):

        '''Event handler for a change in the selected plot type.

        '''

        print('var change handler')
        new_time = self._refresh_times(new_val)
        for p1 in self.plots:
            # different variables have different times available, soneed to
            # set time when selecting a variable
            p1.current_time = new_time
            p1.set_var(new_val)

    def on_region_change(self, attr1, old_val, new_val):

        '''Event handler for a change in the selected plot region.

        '''

        print('region change handler')

        for p1 in self.plots:
            p1.set_region(new_val)

    def on_config_change(self, plot_index, attr1, old_val, new_val):

        '''Event handler for a change in the selected model configuration output.

        '''

        print('config change handler')
        
        self.plots[plot_index].set_config(new_val)