import datetime
import bokeh.models
import bokeh.layouts

import forest.control

import simim_sat_data

class SimimSatControl(object):
    
    def __init__(self,
                 datasets,
                 init_time,
                 init_fcast_time,
                 init_wavelength,
                 plot_list,
                 bokeh_imgs,
                 colorbar_widget):
        
        '''
        
        '''
        self.data_time_dd = None

        self.datasets = datasets
        self.current_fcast_time = init_fcast_time
        self.plot_list = plot_list
        self.bokeh_imgs = bokeh_imgs
        self.colorbar_div = colorbar_widget
        self.current_time = init_time
        self.wavelengths_list = ['W', 'I', 'V']
        self.dd_label_dict = {'W': u'Water vapour (6.2\u03BCm)',
                              'I': u'Infra-red (10.4\u03BCm)', 
                              'V': u'Visible (0.64\u03BCm)'}

        self.current_wavelength = init_wavelength
        self._refresh_times(update_gui=False)
        self.create_widgets()

        self.process_events = True

    def __str__(self):
        
        '''
        
        '''
        
        pass

    def create_widgets(self):

        '''
        
        '''
        
        # Create wavelength selection dropdown widget
        wl_dd_list = [(self.dd_label_dict[k1], k1) for 
                      k1 in self.dd_label_dict.keys()]
        self.wavelength_dd = bokeh.models.widgets.Dropdown(label='Wavelength',
                                                           menu=wl_dd_list,
                                                           button_type='warning',
                                                           value=self.current_wavelength)
        self.wavelength_dd.on_change('value', self.on_type_change)

        # Create previous timestep button widget
        self.time_prev_button = bokeh.models.widgets.Button(label='Prev',
                                                            button_type='warning',
                                                            width=100)
        self.time_prev_button.on_click(self.on_time_prev)
        
        # Create time selection dropdown widget
        time_dd_list = [(k1 + 'UTC', k1) for k1 in self.time_list]
        self.data_time_dd = bokeh.models.widgets.Dropdown(label='Time',
                                                          menu=time_dd_list,
                                                          button_type='warning',
                                                          width=300,
                                                          value=self.current_time)
        self.data_time_dd.on_change('value', self.on_data_time_change)

        # Create next timestep button widget
        self.time_next_button = bokeh.models.widgets.Button(label='Next',
                                                            button_type='warning',
                                                            width=100)
        self.time_next_button.on_click(self.on_time_next)

        model_run_list = forest.control.create_model_run_list(self.datasets)
        self.model_run_dd = \
            bokeh.models.widgets.Dropdown(label='Model run',
                                          menu=model_run_list,
                                          button_type='warning',
                                          value=self.current_fcast_time,)
        self.model_run_dd.on_change('value', self._on_model_run_change)

        # Set layout rows for widgets
        self.time_row = \
            bokeh.layouts.row(self.time_prev_button,
                              bokeh.models.Spacer(width=20, height=60),
                              self.data_time_dd,
                              bokeh.models.Spacer(width=20, height=60),
                              self.time_next_button,
                              bokeh.models.Spacer(width=20, height=60),
                              self.model_run_dd)
        self.major_config_row = bokeh.layouts.row(self.wavelength_dd)
        self.plots_row = bokeh.layouts.row(*self.bokeh_imgs)
        self.info_row = bokeh.layouts.row(bokeh.models.Spacer(width=400, height=100), 
                                          self.colorbar_div,
                                          bokeh.models.Spacer(width=400, height=100))
        
        # Create main layout
        self.main_layout = bokeh.layouts.column(self.time_row,
                                                self.major_config_row,
                                                self.plots_row,
                                                self.info_row,
                                               )

    def on_data_time_change(self, attr1, old_val, new_val):
        
        '''Event handler for a change in the selected forecast data time.
        
        '''
        if not self.process_events:
            return
        print('selected new time {0}'.format(new_val))
        
        self.current_time = new_val
        for p1 in self.plot_list:
            p1.set_data_time(self.current_time)

    def on_time_prev(self):
        
        '''Event handler for changing to previous time step
        
        '''
        if not self.process_events:
            return

        print('selected previous time step')   
        current_index = self.time_list.index(self.current_time + 'UTC')
        if current_index > 0:
            self.current_time = self.time_list[current_index - 1][:-3]
            for p1 in self.plot_list:
                p1.set_data_time(self.current_time)  
        else:
            print('No previous time step')
                
    def on_time_next(self):
        
        '''
        
        '''
        if not self.process_events:
            return

        print('selected next time step')     
        current_index = self.time_list.index(self.current_time + 'UTC')
        if current_index < len(self.time_list) - 1:
            self.current_time = self.time_list[current_index + 1][:-3]
            for p1 in self.plot_list:
                p1.set_data_time(self.current_time)  
        else:
            print('No next time step')

    def _refresh_times(self, update_gui=True):
        """
        Update list of times available from current data type
        :return:
        """
        # Update time dropdown menu with times for new variable

        current_data_simim = self.datasets[self.current_fcast_time][simim_sat_data.SIMIM_KEY]['data']
        current_times_simim = current_data_simim.get_times(self.current_wavelength)
        current_data_sat = self.datasets[self.current_fcast_time][simim_sat_data.HIMAWARI8_KEY]['data']
        current_times_sat = current_data_sat.get_times(self.current_wavelength)

        self.time_list = [time_str for time_str in current_times_simim if time_str in current_times_simim]
        self.time_list.sort()

        if update_gui and self.data_time_dd:
            self.data_time_dd.menu = [(k1+'UTC', k1) for k1 in self.time_list]
        if self.current_time not in self.time_list:
            self.current_time = self.time_list[0]
            if update_gui and self.data_time_dd:
                self.process_events = False
                self.data_time_dd.value = self.current_time
                self.process_events = True

    def on_type_change(self, attr1, old_val, new_val):

        '''Event handler for a change in the selected plot type.
        
        '''
        if not self.process_events:
            return

        print('selected new wavelength {0}'.format(new_val))
        self.current_wavelength = new_val
        
        self._refresh_times(update_gui=True)

        for p1 in self.plot_list:
            p1.set_var(self.current_wavelength)

    def _on_model_run_change(self, attr1, old_val, new_val):
        if not self.process_events:
            return

        print('new model run selected: {0}'.format(new_val))
        self.current_fcast_time = new_val
        self._refresh_times(update_gui=True)

        for p1 in self.plot_list:
            p1.current_time = self.current_time
            p1.set_dataset(self.datasets[self.current_fcast_time],
                           self.current_fcast_time,
                           )