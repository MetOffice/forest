"""Module containing a class to manage the widgets for Sim Im vs Him-8. 

Functions
---------

None

Classes
-------

- SimimSatControl -- Main class for defining Model vs GPM widgets.

"""

import datetime
import bokeh.models
import bokeh.layouts


class SimimSatControl(object):
 
    """Main widget configuration class.
    
    Methods
    -------
    
    - __init__() -- Factory method.
    - __str__() -- String method.
    - create_widgets() -- Configure widgets and widget layout.
    - on_time_prev() -- Event handler for changing to prev time step.
    - on_data_tine_change() -- Event handler for a change in data time.
    - on_time_next() -- Event handler for changing to next time step.
    - on_type_change() -- Event handler for a change in wavelength.
    
    Attributes
    ----------
    
    - datasets -- Dict; Dict of Forest datasets.
    - plot_list -- List; List of ForestPlot objects.
    - bokeh_imgs -- List; List of bokeh image objects.
    - colorbar_div -- bokeh.models.widgets.Div; C'bar widget object.
    - current_time -- Int; Index of current time step.
    - fcast_time_obj -- datetime object; Date/time of model run.
    - wavelengths_list -- List; Single-letter keys for wavelengths.
    - dd_label_dict -- Dict; Dropdown labels for wavelengths.
    - time_list -- List; Time strings of current wavelength data times.
    - time_prev_button -- bokeh.models.widgets.Button; Prev button.
    - data_time_slider -- bokeh.models.widgets.Slider; Time slider.
    - time_next_button -- bokeh.models.widgets.Button; Next button.
    - wavelength_dd -- bokeh.models.widgets.Dropdown; Model dropdown.
    - time_row -- bokeh.layouts.row object; Set time row.
    - major_config_row -- bokeh.layouts.row object; Set mjr config row.
    - plots_row -- bokeh.layouts.row object; Set plots row.
    - info_row -- bokeh.layouts.row object; Set info row.
    - main_layout -- bokeh.layouts.column object; Set row layout.
    
    """

    def __init__(self, datasets, init_time, fcast_time_obj, 
                 plot_list, bokeh_imgs, colorbar_widget):
        
        """Initialisation function for ForestController class."""
        
        self.datasets = datasets
        self.plot_list = plot_list
        self.bokeh_imgs = bokeh_imgs
        self.colorbar_div = colorbar_widget
        self.current_time = init_time
        self.fcast_time_obj = fcast_time_obj
        self.wavelengths_list = ['W', 'I', 'V']
        self.dd_label_dict = {'W': u'Water vapour (6.2\u03BCm)',
                              'I': u'Infra-red (10.4\u03BCm)', 
                              'V': u'Visible (0.64\u03BCm)'}

        self.create_widgets()

    def __str__(self):
        
        """Return string"""
        
        return "SimimSatControl object"

    def create_widgets(self):

        """Configure widgets and widget layout."""

        # Create previous timestep button widget
        self.time_prev_button = bokeh.models.widgets.Button(label='Prev',
                                                            button_type='warning',
                                                            width=100)
        self.time_prev_button.on_click(self.on_time_prev)
        
        # Create time selection dropdown widget
        self.time_list = sorted([time_str + 'UTC' for time_str in 
                                 self.datasets['simim']['data'].get_data('I').keys()
                                 if time_str in self.datasets['simim']['data'].get_data('I').keys()])
        time_dd_list = [(k1, k1) for k1 in self.time_list]
        self.data_time_dd = bokeh.models.widgets.Dropdown(label='Time',
                                                          menu=time_dd_list,
                                                          button_type='warning',
                                                          width=300)
        self.data_time_dd.on_change('value', self.on_data_time_change)

        # Create next timestep button widget
        self.time_next_button = bokeh.models.widgets.Button(label='Next',
                                                            button_type='warning',
                                                            width=100)
        self.time_next_button.on_click(self.on_time_next)
        
        # Create wavelength selection dropdown widget
        wl_dd_list = [(self.dd_label_dict[k1], k1) for 
                      k1 in self.dd_label_dict.keys()]
        self.wavelength_dd = bokeh.models.widgets.Dropdown(label='Wavelength',
                                                           menu=wl_dd_list,
                                                           button_type='warning')
        self.wavelength_dd.on_change('value', self.on_type_change)
        
        # Set layout rows for widgets
        self.time_row = \
            bokeh.layouts.row(self.time_prev_button,
                              bokeh.models.Spacer(width=20, height=60),
                              self.data_time_dd,
                              bokeh.models.Spacer(width=20, height=60),
                              self.time_next_button)
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

    def on_time_prev(self):
        
        """Event handler for changing to previous time step."""
        
        print('selected previous time step')   
        current_index = self.time_list.index(self.current_time + 'UTC')
        if current_index > 0:
            self.current_time = self.time_list[current_index - 1][:-3]
            for p1 in self.plot_list:
                p1.set_data_time(self.current_time)  
        else:
            print('No previous time step')
            
    def on_data_time_change(self, attr1, old_val, new_val):
        
        """Event handler for a change in the selected forecast data time.

        Arguments
        ---------
        
        - attr1 -- Str; Attribute of dropdown which changes.
        - old_val -- Int; Old value from dropdown.
        - new_val -- Int; New value from dropdown.

        """
        
        print('selected new time {0}'.format(new_val))
        
        self.current_time = new_val[:-3]
        for p1 in self.plot_list:
            p1.set_data_time(self.current_time)
                
    def on_time_next(self):
        
        """Event handler for changing to next time step."""
        
        print('selected next time step')     
        current_index = self.time_list.index(self.current_time + 'UTC')
        if current_index < len(self.time_list) - 1:
            self.current_time = self.time_list[current_index + 1][:-3]
            for p1 in self.plot_list:
                p1.set_data_time(self.current_time)  
        else:
            print('No next time step')

    def on_type_change(self, attr1, old_val, new_val):

        """Event handler for a change in the selected wavelength.

        Arguments
        ---------
        
        - attr1 -- Str; Attribute of dropdown which changes.
        - old_val -- Int; Old value from dropdown.
        - new_val -- Int; New value from dropdown.

        """

        print('selected new wavelength {0}'.format(new_val))
        current_type = new_val
        
        # Update time dropdown menu with times for new variable
        self.time_list = sorted([time_str + 'UTC' for time_str in 
                                 self.datasets['simim']['data'].get_data(new_val).keys()
                                 if time_str in self.datasets['simim']['data'].get_data(new_val).keys()])

        self.data_time_dd.menu = [(k1, k1) for k1 in self.time_list]

        for p1 in self.plot_list:
            p1.set_var(current_type)