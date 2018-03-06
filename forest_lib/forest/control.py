"""Module containing a class to manage the widgets for Forest. 

Functions
---------

- create_dropdown_opt_list() -- Used to set dropdown labels.
- create_dropdown_opt_list_from dict() -- Used to set dropdown labels.

Classes
-------

- ForestController -- Main class for defining Forest widgets.

"""


import functools
import threading
import bokeh.models.widgets
import bokeh.layouts
import bokeh.plotting
import tornado

MODEL_DD_DICT = {'n1280_ga6': '"Global" GA6 10km', 
                 'km4p4_ra1t': 'SE Asia 4.4km', 
                 'indon2km1p5_ra1t': 'Indonesia 1.5km', 
                 'mal2km1p5_ra1t': 'Malaysia 1.5km', 
                 'phi2km1p5_ra1t': 'Philippines 1.5km'}

VARIABLE_DD_DICT = {'precipitation': 'Precipitation', 
                    'air_temperature': 'Air Temperature', 
                    'wind_vectors': 'Wind vectors', 
                    'wind_mslp': 'Wind and MSLP', 
                    'wind_streams': 'Wind streamlines', 
                    'mslp': 'MSLP', 
                    'cloud_fraction': 'Cloud fraction'}

REGION_DD_DICT = {'indonesia': 'Indonesia', 
                  'malaysia': 'Malaysia', 
                  'phillipines': 'Philippines', 
                  'se_asia': 'SE Asia'}


def create_dropdown_opt_list(iterable1):
    
    """Return list of 2-tuples with matching values. 
    
    Used for creating dropdown menu labels which match return values.
    
    Arguments
    ---------
    
    - iterable1 -- Iterable; Used to set 2-tuple values.
    
    """
    
    return [(k1, k1) for k1 in iterable1]


def create_dropdown_opt_list_from_dict(dict1, iterable1):
    
    """Create list of 2-tuples with values from dictionary.
    
    Used for creating descriptive dropdown menu labels which do not 
    match return values.
    
    Arguments
    ---------
    
    - dict1 -- Dict; Used to replace iterable if iterable value == key.
    - iterable -- Iterable; Used to set 2-tuple values.
    
    """
    
    dd_tuple_list = [(dict1[k1], k1) if k1 in dict1.keys()
                     else (k1, k1) for k1 in iterable1]

    return dd_tuple_list


class ForestController(object):

    """Main widget configuration class.
    
    Methods
    -------
    
    - __init__() -- Factory method.
    - create_widgets() -- Configure widgets and widget layout.
    - on_time_prev() -- Event handler for changing to prev time step.
    - on_data_tine_change() -- Event handler for a change in data time.
    - on_time_next() -- Event handler for changing to next time step.
    - on_var_change() -- Event handler for a change in the plot var.
    - on_region_change() -- Event handler for a change in plot region.
    - on_config_change() -- Event handler for a change in the config.
    
    Attributes
    ----------
   
    - plots -- List; List of ForestPlot objects.
    - bokeh_imgs -- List; List of bokeh image objects.
    - init_time -- Int; Index of initial time step.
    - num_times -- Int; Number of data time steps available.
    - region_dict -- Dict; Dict of regions and their lat/lon bounds.
    - plot_names -- List; List of plot names
    - datasets -- Dict; Dict of Forest datasets.
    - colorbar_div -- bokeh.models.widgets.Div; C'bar div widget.
    - stats_widgets -- bokeh.models.widgets.Div; Stats div widget.
    - bokeh_doc -- bokeh curdoc object;
    - time_prev_button -- bokeh.models.widgets.Button; Prev button.
    - data_time_slider -- bokeh.models.widgets.Slider; Time slider.
    - time_next_button -- bokeh.models.widgets.Button; Next button.
    - model_var_dd -- bokeh.models.widgets.Dropdown; Var dropdown.
    - region_dd -- bokeh.models.widgets.Dropdown; Region dropdown.
    - left_model_dd -- bokeh.models.widgets.Dropdown; Model dropdown.
    - right_model_dd -- bokeh.models.widgets.Dropdown; Model dropdown.
    - time_row -- bokeh.layouts.row object; Set time row.
    - major_config_row -- bokeh.layouts.row object; Set mjr config row.
    - minor_config_row -- bokeh.layouts.row object; Set mnr config row.
    - plots_row -- bokeh.layouts.row object; Set plots row.
    - info_row -- bokeh.layouts.row object; Set info row.
    - main_layout -- bokeh.layouts.column object; Set row layout.
    
    """

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

        """Initialisation function for ForestController class."""

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

        """Configure widgets and widget layout."""

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
                                        value=self.init_time,
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

        # Create variable selection dropdown widget
        variable_menu_list = \
            create_dropdown_opt_list_from_dict(VARIABLE_DD_DICT,
                                               self.plot_names)
        self.model_var_dd = \
            bokeh.models.widgets.Dropdown(label='Variable to visualise',
                                          menu=variable_menu_list,
                                          button_type='warning')
        self.model_var_dd.on_change('value', self.on_var_change)
        
        # Create region selection dropdown menu region
        region_menu_list = \
            create_dropdown_opt_list_from_dict(REGION_DD_DICT,
                                               self.region_dict.keys())
        self.region_dd = bokeh.models.widgets.Dropdown(menu=region_menu_list,
                                                       label='Region',
                                                       button_type='warning')
        self.region_dd.on_change('value', self.on_region_change)

        # Create left figure model selection dropdown menu widget
        dataset_menu_list = create_dropdown_opt_list_from_dict(MODEL_DD_DICT,
                                                               self.datasets.keys())
        self.left_model_dd = \
            bokeh.models.widgets.Dropdown(menu=dataset_menu_list,
                                          label='Left display',
                                          button_type='warning')
        self.left_model_dd.on_change('value',
                                     functools.partial(self.on_config_change,
                                                       0))
        # Create right figure model selection dropdown menu widget
        self.right_model_dd = \
            bokeh.models.widgets.Dropdown(menu=dataset_menu_list,
                                          label='Right display',
                                          button_type='warning')
        self.right_model_dd.on_change('value',
                                      functools.partial(self.on_config_change,
                                                        1))

        # Layout widgets
        self.time_row = \
            bokeh.layouts.row(self.time_prev_button,
                              bokeh.models.Spacer(width=20, height=60),
                              self.data_time_slider,
                              bokeh.models.Spacer(width=20, height=60),
                              self.time_next_button)
        self.major_config_row = bokeh.layouts.row(self.model_var_dd, 
                                                  self.region_dd)
        self.minor_config_row = bokeh.layouts.row(self.left_model_dd, 
                                                  self.right_model_dd)
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
        
        """Event handler for changing to previous time step."""
        
        print('selected previous time step')
        
        current_time = int(self.data_time_slider.value - 1)
        if current_time >= 0:
            self.data_time_slider.value = current_time
        else:
            print('Cannot select time < 0')
      
    def on_data_time_change(self, attr1, old_val, new_val):

        """Event handler for a change in the selected forecast data time.

        Arguments
        ---------
        
        - attr1 -- Str; Attribute of slider which changes.
        - old_val -- Int; Old value from slider.
        - new_val -- Int; New value from slider.

        """

        print('data time handler')
        
        for p1 in self.plots:
            p1.set_data_time(new_val)

    def on_time_next(self):
        
        """Event handler for changing to next time step."""
        
        print('selected next time step')
        
        current_time = int(self.data_time_slider.value + 1)
        if current_time < self.num_times:
            self.data_time_slider.value = current_time
        else:
            print('Cannot select time > num_times')            
            
    def on_var_change(self, attr1, old_val, new_val):

        """Event handler for a change in the selected plot variable.

        Arguments
        ---------
        
        - attr1 -- Str; Attribute of dropdown which changes.
        - old_val -- Int; Old value from dropdown.
        - new_val -- Int; New value from dropdown.

        """

        print('var change handler')

        for p1 in self.plots:
            p1.set_var(new_val)

    def on_region_change(self, attr1, old_val, new_val):

        """Event handler for a change in the selected plot region.

        Arguments
        ---------
        
        - attr1 -- Str; Attribute of dropdown which changes.
        - old_val -- Int; Old value from dropdown.
        - new_val -- Int; New value from dropdown.

        """

        print('region change handler')

        for p1 in self.plots:
            p1.set_region(new_val)

    def on_config_change(self, plot_index, attr1, old_val, new_val):

        """Event handler for a change in the selected data configuration.

        Arguments
        ---------
        
        - plot_index -- Int; Selects which plot to change config for.
        - attr1 -- Str; Attribute of dropdown which changes.
        - old_val -- Int; Old value from dropdown.
        - new_val -- Int; New value from dropdown.

        """

        print('config change handler')
        
        self.plots[plot_index].set_config(new_val)