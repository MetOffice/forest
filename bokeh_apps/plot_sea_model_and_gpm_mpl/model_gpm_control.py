"""Module containing a class to manage the widgets for Model vs GPM. 

Functions
---------

- create_dropdown_opt_list_from dict() -- Used to set dropdown labels.

Classes
-------

- ModelGpmControl -- Main class for defining Model vs GPM widgets.

"""

import functools
import bokeh
import bokeh.model
import bokeh.layouts


MODEL_DD_DICT = {'n1280_ga6': '"Global" GA6 10km', 
                 'km4p4_ra1t': 'SE Asia 4.4km', 
                 'indon2km1p5_ra1t': 'Indonesia 1.5km', 
                 'mal2km1p5_ra1t': 'Malaysia 1.5km', 
                 'phi2km1p5_ra1t': 'Philippines 1.5km'}


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


class ModelGpmControl(object):
    
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
    
    - datasets -- Dict; Dict of Forest datasets.
    - init_time -- Int; Index of initial time step.
    - num_times -- Int; Number of data time steps available.
    - plot_list -- List; List of ForestPlot objects.
    - bokeh_img_list -- List; List of bokeh image objects.
    - time_prev_button -- bokeh.models.widgets.Button; Prev button.
    - data_time_slider -- bokeh.models.widgets.Slider; Time slider.
    - time_next_button -- bokeh.models.widgets.Button; Next button.
    - model_dd -- bokeh.models.widgets.Dropdown; Model dropdown.
    - imerg_rbg -- bokeh.models.widgets.RadioButtonGroup; IMERG RBG.
    - accum_div -- bokeh.models.widgets.Div; Accum label div.
    - accum_rbg -- bokeh.models.widgets.RadioButtonGroup; ACCUM RBG.
    - time_row -- bokeh.layouts.row object; Set time row.
    - major_config_row -- bokeh.layouts.row object; Set mjr config row.
    - minor_config_row -- bokeh.layouts.row object; Set mnr config row.
    - plots_row -- bokeh.layouts.row object; Set plots row.
    - info_row -- bokeh.layouts.row object; Set info row.
    - main_layout -- bokeh.layouts.column object; Set row layout.
    
    """
    
    def __init__(self,
                 datasets,
                 init_time,
                 num_times,
                 plot_list,
                 bokeh_img_list,
                 ):
        
        """Initialisation function for ForestController class."""

        
        self.datasets = datasets
        self.init_time = init_time
        self.num_times = num_times
        self.plot_list = plot_list
        self.bokeh_img_list = bokeh_img_list
        self.create_widgets()

    def __str__(self):
        
        """Return string"""
        
        return 'MVC-style controller class for model rainfall vs GPM app.'

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
                                        step=3,
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

    def on_time_prev(self):
        
        """Event handler for changing to previous time step."""
        
        print('selected previous time step')      
        
        time_step = int(self.accum_rbg.labels[self.accum_rbg.active][:-2])
        current_time = int(self.data_time_slider.value - time_step)
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
        
        print('selected new time {0}'.format(new_val))
        
        time_step = int(self.accum_rbg.labels[self.accum_rbg.active][:-2])
        current_time = int(new_val / time_step)
        for p1 in self.plot_list:
            p1.set_data_time(current_time)

    def on_time_next(self):
        
        """Event handler for changing to next time step."""

        print('selected next time step')   
        
        time_step = int(self.accum_rbg.labels[self.accum_rbg.active][:-2])
        current_time = int(self.data_time_slider.value + time_step)
        if current_time < self.num_times:
            self.data_time_slider.value = current_time
        else:
            print('Cannot select time > num_times')        

    def on_config_change(self, plot_index, attr1, old_val, new_val):

        """Event handler for a change in the selected data configuration.

        Arguments
        ---------
        
        - plot_index -- Int; Selects which plot to change config for.
        - attr1 -- Str; Attribute of dropdown which changes.
        - old_val -- Int; Old value from dropdown.
        - new_val -- Int; New value from dropdown.

        """
        
        print('selected new config {0}'.format(new_val))
        self.plot_list[plot_index].set_config(new_val)

    def on_imerg_change(self, plot_index, attr1, old_val, new_val):
        
        """Event handler for a change in the selected IMERG type.

        Arguments
        ---------
        
        - plot_index -- Int; Selects which plot to change config for.
        - attr1 -- Str; Attribute of dropdown which changes.
        - old_val -- Int; Old value from dropdown.
        - new_val -- Int; New value from dropdown.

        """
        
        imerg_list = [ds_name for ds_name in self.datasets.keys()
                      if 'imerg' in ds_name]
        print('selected new config {0}'.format(imerg_list[new_val]))
        new_config = imerg_list[new_val]
        self.plot_list[plot_index].set_config(new_config)
        
    def on_accum_change(self, plot_index, attr1, old_val, new_val):
        
        """Event handler for a change in the selected precip accum.

        Arguments
        ---------
        
        - plot_index -- Int; Selects which plot to change config for.
        - attr1 -- Str; Attribute of dropdown which changes.
        - old_val -- Int; Old value from dropdown.
        - new_val -- Int; New value from dropdown.

        """
        
        # Change slider step based on new accumulation range
        self.data_time_slider.step = int(self.accum_rbg.labels[new_val][:-2])
        print('selected new accumulation time span {0}'.format(self.accum_rbg.labels[new_val]))
        new_var = 'accum_precip_{0}'.format(self.accum_rbg.labels[new_val])
        for plot in self.plot_list:
            plot.set_var(new_var)