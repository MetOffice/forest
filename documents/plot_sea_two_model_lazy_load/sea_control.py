import functools
import threading

import bokeh.models.widgets
import bokeh.layouts
import bokeh.plotting

import tornado

# set up bokeh widgets
def create_dropdown_opt_list(iterable1):
    return [(k1,k1) for k1 in iterable1]

class SEA_controller(object):
    """
    """
    def __init__(self, init_time, num_times, datasets, plot_names, plots, bokeh_imgs, stats_widgets, region_dict, bokeh_doc):
        """
        """
        self.plots = plots
        self.bokeh_imgs = bokeh_imgs
        self.init_time = init_time
        self.num_times = num_times
        self.region_dict = region_dict
        self.plot_names = plot_names
        self.datasets = datasets
        self.stats_widgets = stats_widgets
        self.create_widgets()
        self.bokeh_doc = bokeh_doc
    
    def create_widgets(self):
        """
        """
        self.model_var_list_desc = 'Attribute to visualise'

        self.model_var_dd = \
            bokeh.models.widgets.Dropdown(label=self.model_var_list_desc,
                                        menu=create_dropdown_opt_list(self.plot_names),
                                        button_type='warning')
        self.model_var_dd.on_change('value',self.on_var_change)
        
            
        self.data_time_slider = bokeh.models.widgets.Slider(start=0, 
                                                    end=self.num_times, 
                                                    value=self.init_time, 
                                                    step=1, 
                                                    title="Data time")
                                                    
        self.data_time_slider.on_change('value',self.on_data_time_change)

        self.region_desc = 'Region'

        region_menu_list = create_dropdown_opt_list(self.region_dict.keys())
        self.region_dd = bokeh.models.widgets.Dropdown(menu=region_menu_list, 
                                                label=self.region_desc,
                                                button_type='warning')
        self.region_dd.on_change('value', self.on_region_change)

        dataset_menu_list = create_dropdown_opt_list(self.datasets.keys())
        left_model_desc = 'Left display'

        self.left_model_dd = bokeh.models.widgets.Dropdown(menu=dataset_menu_list,
                                                    label=left_model_desc,
                                                    button_type='warning')
        self.left_model_dd.on_change('value', self.on_config_change,)


        right_model_desc = 'Right display'
        self.right_model_dd = bokeh.models.widgets.Dropdown(menu=dataset_menu_list, 
                                                    label=right_model_desc,
                                                    button_type='warning')
        self.right_model_dd.on_change('value', self.on_config_change)

        # layout widgets
        self.param_row = bokeh.layouts.row(self.model_var_dd, self.region_dd)
        self.slider_row = bokeh.layouts.row(self.data_time_slider)
        self.config_row = bokeh.layouts.row(self.left_model_dd, self.right_model_dd)

        self.plots_row = bokeh.layouts.row(*self.bokeh_imgs)
        self.stats_row = bokeh.layouts.row(*self.stats_widgets)
        self.main_layout = bokeh.layouts.column(self.param_row, 
                                        self.slider_row,
                                        self.config_row,
                                        self.plots_row,
                                        self.stats_row,
                                        ) 
                                        
    def on_data_time_change(self, attr1, old_val, new_val):
        '''
        Event handler for a change in the selected forecast data time.
        '''
        print('data time handler')
        for p1 in self.plots:
            p1.set_data_time(new_val)        
        self._update_bokeh_plot()
        
    def data_time_async(self, new_val):
        print('data_time_async')
        for p1 in self.plots:
            p1.set_data_time(new_val)        
        self.bokeh_doc.add_next_tick_callback(self._update_bokeh_plot)        
        print('data_time_async')

    def on_var_change(self, attr1, old_val, new_val):
        '''
        Event handler for a change in the selected plot type.
        '''
        print('var change handler')
        for p1 in self.plots:
            p1.set_var(new_val)        
        self._update_bokeh_plot()
        
    def var_change_async(self, new_val):
        print('var_change_async')
        for p1 in self.plots:
            p1.set_var(new_val)
        self.bokeh_doc.add_next_tick_callback(self._update_bokeh_plot)            
        print('var_change_async')

    def on_region_change(self, attr1, old_val, new_val):
        '''
        Event handler for a change in the selected plot region.
        '''
        print('region change handler')
        for p1 in self.plots:
            p1.set_region(new_val)        
        self._update_bokeh_plot()            
        
    def region_change_async(self, new_val):
        print('region_change_async')
    
        for p1 in self.plots:
            p1.set_region(new_val)
        self.bokeh_doc.add_next_tick_callback(self._update_bokeh_plot)            
        print('region_change_async')
        
    
    def on_config_change(self, attr1, old_val, new_val):
        '''
        Event handler for a change in the selected model configuration output.
        '''
        print('config change handler')
        for p1 in self.plots:
            p1.set_config(new_val)        
        self._update_bokeh_plot()       
        
    def config_change_async(self, new_val):
        print('config_change_async')
        for p1 in self.plots:
            p1.set_config(new_val)
        self.bokeh_doc.add_next_tick_callback(self._update_bokeh_plot)            
        print('config_change_async')
                
    def _update_bokeh_plot(self):
        print('updating bokeh plot')
        for p1 in self.plots:
            p1.update_bokeh_img_plot_from_fig()
    