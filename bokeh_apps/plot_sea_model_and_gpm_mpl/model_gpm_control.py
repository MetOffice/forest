import functools

import bokeh
import bokeh.model
import bokeh.layouts

import forest.data

MODEL_DD_DICT = {'n1280_ga6': '"Global" GA6 10km', 
                 'km4p4_ra1t': 'SE Asia 4.4km', 
                 'indon2km1p5_ra1t': 'Indonesia 1.5km', 
                 'mal2km1p5_ra1t': 'Malaysia 1.5km', 
                 'phi2km1p5_ra1t': 'Philippines 1.5km'}


def create_dropdown_opt_list_from_dict(dict1, iterable1):
    
    '''Create list of 2-tuples from dictionary for creating dropdown 
        menu labels.
    
    '''
    
    dd_tuple_list = [(dict1[k1], k1) if k1 in dict1.keys()
                     else (k1, k1) for k1 in iterable1]

    return dd_tuple_list


class ModelGpmControl(object):
    
    '''
    
    '''
    
    def __init__(self,
                 datasets,
                 init_time_ix,
                 available_times,
                 plot_list,
                 bokeh_img_list,
                 ):
        
        '''
        
        '''
        
        self.datasets = datasets
        self.available_times = available_times
        self.init_time_ix = init_time_ix
        self.init_time = self.available_times[self.init_time_ix]
        self.num_times = self.available_times.shape[0]
        self.plot_list = plot_list
        self.bokeh_img_list = bokeh_img_list
        self.create_widgets()

    def __str__(self):
        
        '''
        
        '''
        
        return 'MVC-style controller class for model rainfall vs GPM app.'

    def create_widgets(self):
        
        '''Set up bokeh widgets
        
        '''
        
        def create_dropdown_opt_list(iterable1):
            return [(k1, k1) for k1 in iterable1]

        # Create previous timestep button widget
        self.time_prev_button = \
            bokeh.models.widgets.Button(label='Prev',
                                        button_type='warning',
                                        width=100)
        self.time_prev_button.on_click(self.on_time_prev)
        
        # Create time selection slider widget
        self.data_time_slider = \
            bokeh.models.widgets.Slider(start=0,
                                        end=self.num_times,
                                        value=self.init_time_ix,
                                        step=1,
                                        title="Data time",
                                        width=400)
        self.data_time_slider.on_change('value', self.on_data_time_change)

        # Create next timestep button widget
        self.time_next_button = \
            bokeh.models.widgets.Button(label='Next',
                                        button_type='warning',
                                        width=100)
        self.time_next_button.on_click(self.on_time_next)
        
        # Create model selection dropdown menu widget
        model_list = [ds_name for ds_name in self.datasets.keys()
                      if 'imerg' not in ds_name]
        model_menu_list = create_dropdown_opt_list_from_dict(MODEL_DD_DICT,
                                                             model_list)
        self.model_dd = bokeh.models.widgets.Dropdown(menu=model_menu_list,
                                                      label= 'Model display',
                                                      button_type='warning',
                                                      width=400)
        self.model_dd.on_change('value', 
                                functools.partial(self.on_config_change, 0))

        # Create GPM IMERG selection radio button group widget
        imerg_labels = [ds_name for ds_name in self.datasets.keys() if 'imerg' in ds_name]
        self.imerg_rbg = \
            bokeh.models.widgets.RadioButtonGroup(labels=imerg_labels,
                                                  button_type='warning',
                                                  width=800)
        self.imerg_rbg.on_change('active', 
                                 functools.partial(self.on_imerg_change, 1))

        # Create accumulation selection label div widget
        accum_div_text = '<p style="vertical-align:middle;">Accumulation window:</p>'
        self.accum_div = bokeh.models.widgets.Div(text=accum_div_text,
                                                  height=60,
                                                  width=150)
        
        # Create accumulation selection radio button group widget
        accum_labels = ['{}hr'.format(val) for val in [3, 6, 12, 24]]
        self.accum_rbg = \
            bokeh.models.widgets.RadioButtonGroup(labels=accum_labels,
                                                  button_type='warning',
                                                  width=800,
                                                  active=0)
        self.accum_rbg.on_change('active', 
                                 functools.partial(self.on_accum_change, 1))
        
        # Create colorbar div widget
        colorbar_link = "<img src='plot_sea_model_and_gpm_mpl/static/" + \
                        "precip_accum_colorbar.png'\>"
        self.colorbar_div = bokeh.models.widgets.Div(text=colorbar_link, 
                                                     width=800, 
                                                     height=100)
        
        # Set layout for widgets
        self.time_row = \
            bokeh.layouts.row(self.time_prev_button,
                              bokeh.models.Spacer(width=20, height=60),
                              self.data_time_slider,
                              bokeh.models.Spacer(width=20, height=60),
                              self.time_next_button)
        self.major_config_row = \
            bokeh.layouts.row(self.model_dd, 
                              bokeh.models.Spacer(width=20, height=60),
                              self.imerg_rbg)
        self.minor_config_row = bokeh.layouts.row(self.accum_div,
                                                  self.accum_rbg)
        self.plots_row = bokeh.layouts.row(*self.bokeh_img_list)
        self.info_row = \
            bokeh.layouts.row(bokeh.models.Spacer(width=400, height=100), 
                              self.colorbar_div,
                              bokeh.models.Spacer(width=400, height=100))

        self.main_layout = bokeh.layouts.column(self.time_row,
                                                self.major_config_row,
                                                self.minor_config_row,
                                                self.plots_row,
                                                self.info_row,
                                               )

    def on_data_time_change(self, attr1, old_val, new_val):
        
        '''Event handler for a change in the selected forecast data time.
        
        '''
        
        print('selected new time {0}'.format(new_val))
        new_time = self.available_times[new_val]
        for p1 in self.plot_list:
            p1.set_data_time(new_time)

    def on_time_prev(self):
        
        '''Event handler for changing to previous time step
        
        '''
        
        print('selected previous time step')       
        current_time = self.data_time_slider.value - 1
        if current_time >= 0:
            self.data_time_slider.value = current_time
        else:
            print('Cannot select time < 0')

    def on_time_next(self):
        
        '''
        
        '''
        
        print('selected next time step')       
        current_time = self.data_time_slider.value + 1
        if current_time < self.num_times:
            self.data_time_slider.value = current_time
        else:
            print('Cannot select time > num_times')

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

    def on_config_change(self, plot_index, attr1, old_val, new_val):
        
        '''Event handler for a change in the selected model configuration output.
        
        '''
        print('selected new config {0}'.format(new_val))
        self.plot_list[plot_index].set_config(new_val)

    def on_imerg_change(self, plot_index, attr1, old_val, new_val):
        
        '''Event handler for a change in the selected model configuration output.
        
        '''
        
        imerg_list = [ds_name for ds_name in self.datasets.keys()
                      if 'imerg' in ds_name]
        print('selected new config {0}'.format(imerg_list[new_val]))
        new_config = imerg_list[new_val]
        self.plot_list[plot_index].set_config(new_config)
        
    def on_accum_change(self, plot_index, attr1, old_val, new_val):
        
        '''Event handler for a change in the selected model configuration output.
        
        '''
        # Change slider step based on new accumulation range
        print('selected new accumulation time span {0}'.format(self.accum_rbg.labels[new_val]))
        new_var = 'accum_precip_{0}'.format(self.accum_rbg.labels[new_val])
        new_time = self._refresh_times(new_var)

        for plot1 in self.plot_list:
            plot1.current_time = new_time
            plot1.set_var(new_var)