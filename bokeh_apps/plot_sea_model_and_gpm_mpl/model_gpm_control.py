import functools

import bokeh
import bokeh.model
import bokeh.layouts

import forest.data
import forest.control

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
                 init_var,
                 datasets,
                 init_time_ix,
                 init_fcast_time,
                 plot_list,
                 bokeh_img_list,
                 stats_list
                 ):
        
        '''
        
        '''
        self.current_var = init_var
        self.current_fcast_time = init_fcast_time
        self.data_time_slider = None

        self.datasets = datasets
        self.current_time_index = init_time_ix
        times = self._refresh_times(update_gui=False)
        self.init_time = self.available_times[self.current_time_index]
        self.num_times = self.available_times.shape[0]
        self.plot_list = plot_list
        self.bokeh_img_list = bokeh_img_list
        self.stats_list = stats_list
        self.process_events = True

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
                                        value=self.current_time_index,
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

        # create model run selection dropdown
        # select model run
        model_run_list = forest.control.create_model_run_list(self.datasets)
        self.model_run_dd = \
            bokeh.models.widgets.Dropdown(label='Model run',
                                          menu=model_run_list,
                                          button_type='warning')

        self.model_run_dd.on_change('value', self._on_model_run_change)

        # Create model selection dropdown menu widget
        model_list = [ds_name for ds_name in self.datasets[self.current_fcast_time].keys()
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
                              self.time_next_button,
                              bokeh.models.Spacer(width=20, height=60),
                              self.model_run_dd,
                              )
        self.major_config_row = \
            bokeh.layouts.row(self.model_dd, 
                              bokeh.models.Spacer(width=20, height=60),
                              self.imerg_rbg)
        self.minor_config_row = bokeh.layouts.row(self.accum_div,
                                                  self.accum_rbg)
        self.plots_row = bokeh.layouts.row(*self.bokeh_img_list)
        self.info_row = \
            bokeh.layouts.row(self.stats_list[0],
                              self.colorbar_div,
                              self.stats_list[1],
                              )

        self.main_layout = bokeh.layouts.column(self.time_row,
                                                self.major_config_row,
                                                self.minor_config_row,
                                                self.plots_row,
                                                self.info_row,
                                               )

    def on_data_time_change(self, attr1, old_val, new_val):
        
        '''Event handler for a change in the selected forecast data time.
        
        '''
        if not self.process_events:
            return
        self.current_time_index = new_val
        print('selected new time {0}'.format(self.current_time_index))
        new_time = self.available_times[self.current_time_index]
        for p1 in self.plot_list:
            p1.set_data_time(new_time)

    def on_time_prev(self):
        
        '''Event handler for changing to previous time step
        
        '''
        if not self.process_events:
            return
        print('selected previous time step')       
        new_time_index = self.current_time_index - 1
        if new_time_index >= 0:
            self.current_time_index = new_time_index
            self.data_time_slider.value = self.current_time_index
        else:
            print('Cannot select time < 0')

    def on_time_next(self):
        
        '''
        
        '''
        
        print('selected next time step')       
        new_time_index = self.current_time_index + 1
        if new_time_index < self.num_times:
            self.current_time_index = new_time_index
            self.data_time_slider.value = self.current_time_index
        else:
            print('Cannot select time > num_times')

    def _refresh_times(self, update_gui):
        self.available_times = forest.data.get_available_times(self.datasets[self.current_fcast_time],
                                                               self.current_var)
        try:
            new_time = self.available_times[self.current_time_index]
        except IndexError:
            self.current_time_index = 0
            new_time = self.available_times[self.current_time_index]
            if update_gui and self.data_time_slider:
                self.process_events = False
                self.data_time_slider.value = self.current_time_index
                self.process_events = True
        if update_gui and self.data_time_slider:
            self.data_time_slider.end = self.available_times.shape[0]

        return new_time

    def on_config_change(self, plot_index, attr1, old_val, new_val):
        
        '''Event handler for a change in the selected model configuration output.
        
        '''
        if not self.process_events:
            return
        print('selected new config {0}'.format(new_val))
        self.plot_list[plot_index].set_config(new_val)

    def on_imerg_change(self, plot_index, attr1, old_val, new_val):
        
        '''Event handler for a change in the selected model configuration output.
        
        '''
        if not self.process_events:
            return

        imerg_list = [ds_name for ds_name in self.datasets.keys()
                      if 'imerg' in ds_name]
        print('selected new config {0}'.format(imerg_list[new_val]))
        new_config = imerg_list[new_val]
        self.plot_list[plot_index].set_config(new_config)
        
    def on_accum_change(self, plot_index, attr1, old_val, new_val):
        
        '''Event handler for a change in the selected model configuration output.
        
        '''
        if not self.process_events:
            return
        # Change slider step based on new accumulation range
        print('selected new accumulation time span {0}'.format(self.accum_rbg.labels[new_val]))
        new_var = 'accum_precip_{0}'.format(self.accum_rbg.labels[new_val])
        self.current_var = new_var
        new_time = self._refresh_times(update_gui=True)

        for plot1 in self.plot_list:
            plot1.current_time = new_time
            plot1.set_var(new_var)

    def _on_model_run_change(self, attr1, old_val, new_val):
        if not self.process_events:
            return

        print('selected new model run {0}'.format(new_val))
        self.current_fcast_time = new_val
        new_time = self._refresh_times(update_gui=True)
        for p1 in self.plot_list:
            # different variables have different times available, soneed to
            # set time when selecting a variable
            p1.current_time = new_time
            p1.set_dataset(self.datasets[self.current_fcast_time],
                           self.current_fcast_time)
