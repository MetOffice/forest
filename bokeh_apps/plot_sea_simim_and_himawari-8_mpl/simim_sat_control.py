import datetime
import bokeh.models
import bokeh.layouts



class SimimSatControl(object):
    def __init__(self, datasets, init_time, fcast_time_obj, 
                 plot_list, bokeh_imgs, colorbar_widget):
        
        '''
        
        '''
        
        self.datasets = datasets
        self.plot_list = plot_list
        self.bokeh_imgs = bokeh_imgs
        self.colorbar_div = colorbar_widget
        self.init_time = init_time
        self.fcast_time_obj = fcast_time_obj
        self.wavelengths_list = ['W', 'I', 'V']

        self.create_widgets()

    def __str__(self):
        
        '''
        
        '''
        
        pass

    def create_widgets(self):

        '''
        
        '''
        
        def create_dropdown_opt_list(iterable1):
            
            '''
            
            '''
            
            return [(k1, k1) for k1 in iterable1]
        
        

        # Create wavelength selection dropdown widget
        self.wavelength_dd = \
            bokeh.models.widgets.Dropdown(label='Wavelength',
                                          menu=create_dropdown_opt_list(self.wavelengths_list),
                                          button_type='warning')
        self.wavelength_dd.on_change('value', self.on_type_change)

        # Create time selection dropdown widget
        time_list = sorted([time_str + 'UTC' for time_str in 
                            self.datasets['simim']['data'].get_data('I').keys()
                            if time_str in self.datasets['simim']['data'].get_data('I').keys()])
        self.data_time_dd = \
            bokeh.models.widgets.Dropdown(label='Time',
                                          menu=create_dropdown_opt_list(time_list),
                                          button_type='warning')
        self.data_time_dd.on_change('value', self.on_data_time_change)

        # Create data slider widget
        start_date = self.fcast_time_obj.date()
        end_date = (start_date + datetime.timedelta(days=1))
        value_date = datetime.datetime.strptime(self.init_time[:8], '%Y%m%d').date()
        self.date_slider = bokeh.models.widgets.sliders.DateSlider(start=start_date,
                                                              end=end_date,
                                                              value=value_date,
                                                              step=86400000,
                                                              title='Select hour')
        self.date_slider.on_change('value', self.on_date_slider_change)

        # Create hour slider widget
        self.hour_slider = bokeh.models.widgets.sliders.Slider(start=0,
                                                               end=21,
                                                               value=12,
                                                               step=3,
                                                               title='Select hour')
        self.hour_slider.on_change('value', self.on_hour_slider_change)

        # Set layout rows for widgets
        self.time_row = bokeh.layouts.row(self.data_time_dd)
        #self.slider_row = bokeh.layouts.row(self.date_slider, self.hour_slider)
        self.major_config_row = bokeh.layouts.row(self.wavelength_dd)
        self.plots_row = bokeh.layouts.row(*self.bokeh_imgs)
        self.colorbar_row = bokeh.layouts.row(bokeh.models.Spacer(width=400, height=100), 
                                              self.colorbar_div,
                                              bokeh.models.Spacer(width=400, height=100))
        
        
        # Create main layout
        self.main_layout = bokeh.layouts.column(self.time_row,
                                                #self.slider_row,
                                                self.major_config_row,
                                                self.plots_row,
                                                self.colorbar_row,
                                               )

    def on_data_time_change(self, attr1, old_val, new_val):
        
        '''Event handler for a change in the selected forecast data time.
        
        '''
        
        print('selected new time {0}'.format(new_val))
        
        current_time = new_val[:-3]
        for p1 in self.plot_list:
            p1.set_data_time(current_time)

    def on_date_slider_change(self, attr1, old_val, new_val):
        
        '''Event handler for a change in the selected forecast data date.
        
        '''
        
        print('selected new date {0}'.format(new_val))
        
        current_time = new_val.strftime('%Y%m%d') + self.plot_list[0].current_time[-4:]
        for p1 in self.plot_list:
            p1.set_data_time(current_time)

    def on_hour_slider_change(self, attr1, old_val, new_val):
        
        '''Event handler for a change in the selected forecast data date.
        
        '''
        
        print('selected new date {0}'.format(new_val))
        
        current_time = self.plot_list[0].current_time[:-4] + '{:02d}00'.format(new_val)
        for p1 in self.plot_list:
            p1.set_data_time(current_time)

    def on_type_change(self, attr1, old_val, new_val):

        '''Event handler for a change in the selected plot type.
        
        '''

        print('selected new var {0}'.format(new_val))

        current_type = new_val
        
        # Update time dropdown menu with times for new variable
        time_list = sorted([time_str + 'UTC' for time_str in 
                    self.datasets['simim']['data'].get_data(new_val).keys()
                    if time_str in self.datasets['simim']['data'].get_data(new_val).keys()])
        self.data_time_dd.menu = [(k1, k1) for k1 in time_list]

        for p1 in self.plot_list:
            p1.set_var(current_type)