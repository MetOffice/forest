import functools
import threading

import bokeh.models.widgets
import bokeh.layouts
import bokeh.plotting

import tornado




def create_dropdown_opt_list(iterable1):
    
    '''Create list of 2-tuples with matching values from list for
        creating dropdown menu labels.
    
    '''
    
    return [(k1, k1) for k1 in iterable1]


def create_dropdown_opt_list_from_dict(dict1):
    
    '''Create list of 2-tuples from dictionary for creating dropdown 
        menu labels.
    
    '''
    
    return [(dict1[k1], k1) for k1 in dict1]


class ForestController(object):

    '''

    '''

    def __init__(self,
                 init_time,
                 num_times,
                 datasets,
                 plot_names,
                 plots,
                 bokeh_imgs,
                 colorbar_widget,
                 stats_widgets,
                 region_dict,
                 bokeh_doc):

        '''
        
        '''

        self.plots = plots
        self.bokeh_imgs = bokeh_imgs
        self.init_time = init_time
        self.num_times = num_times
        self.region_dict = region_dict
        self.plot_names = plot_names
        self.datasets = datasets
        self.colorbar_div = colorbar_widget
        self.stats_widgets = stats_widgets
        self.create_widgets()
        self.bokeh_doc = bokeh_doc

    def create_widgets(self):

        '''
        
        '''

        self.model_var_list_desc = 'Attribute to visualise'

        self.model_var_dd = \
            bokeh.models.widgets.Dropdown(label=self.model_var_list_desc,
                                          menu=create_dropdown_opt_list(
                                              self.plot_names),
                                          button_type='warning')
        self.model_var_dd.on_change('value', self.on_var_change)

        # Create previous timestep button widget
        self.time_prev_button = bokeh.models.widgets.Button(label='Prev',
                                                            button_type='warning',
                                                            width=100)
        self.time_prev_button.on_click(self.on_time_prev)
        
        # Create time selection slider widget
        self.data_time_slider = \
            bokeh.models.widgets.Slider(start=0,
                                        end=self.num_times,
                                        value=self.init_time,
                                        step=1,
                                        title="Data time",
                                        width=400)
        self.data_time_slider.on_change('value', self.on_data_time_change)

        # Create next timestep button widget
        self.time_next_button = bokeh.models.widgets.Button(label='Next',
                                                            button_type='warning',
                                                            width=100)
        self.time_next_button.on_click(self.on_time_next)
        
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
        self.major_config_row = bokeh.layouts.row(self.model_var_dd, self.region_dd)
        self.time_row = bokeh.layouts.row(self.time_prev_button,
                                            self.data_time_slider,
                                            self.time_next_button)
        self.minor_config_row = bokeh.layouts.row(
            self.left_model_dd, self.right_model_dd)
        self.plots_row = bokeh.layouts.row(*self.bokeh_imgs)
        self.info_row = bokeh.layouts.row(self.stats_widgets[0], 
                                          self.colorbar_div,
                                          self.stats_widgets[1])

        self.main_layout = bokeh.layouts.column(self.time_row,
                                                self.major_config_row,
                                                self.minor_config_row,
                                                self.plots_row,
                                                self.info_row,
                                                )

    def on_time_prev(self):
        
        '''Event handler for changing to previous time step
        
        '''
        
        print('selected previous time step')
        
        current_time = int(self.data_time_slider.value - 1)
        if current_time >= 0:
            self.data_time_slider.value = current_time
        else:
            print('Cannot select time < 0')

    def on_time_next(self):
        
        '''
        
        '''
        
        print('selected next time step')
        
        current_time = int(self.data_time_slider.value + 1)
        if current_time < self.num_times:
            self.data_time_slider.value = current_time
        else:
            print('Cannot select time > num_times')      
        
    def on_data_time_change(self, attr1, old_val, new_val):

        '''Event handler for a change in the selected forecast data time.

        '''

        print('data time handler')
        
        for p1 in self.plots:
            p1.set_data_time(new_val)

    def on_var_change(self, attr1, old_val, new_val):

        '''Event handler for a change in the selected plot type.

        '''

        print('var change handler')

        for p1 in self.plots:
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