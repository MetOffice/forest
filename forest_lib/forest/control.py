import os
import functools
import threading
import dateutil.parser

import bokeh.models.widgets
import bokeh.layouts
import bokeh.plotting

import forest.data
import forest.feedback

CONFIG_DIR = os.path.dirname(__file__)
FEEDBACK_CONF_FILENAME = 'feedback_fields.conf'

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
    
    '''Create list of 2-tuples with matching values from list for
        creating dropdown menu labels.
    
    '''
    
    return [(k1, k1) for k1 in iterable1]


def create_dropdown_opt_list_from_dict(dict1, iterable1):
    
    '''Create list of 2-tuples from dictionary for creating dropdown 
        menu labels.
    
    '''
    
    dd_tuple_list = [(dict1[k1], k1) if k1 in dict1.keys()
                     else (k1, k1) for k1 in iterable1]

    return dd_tuple_list


def create_model_run_list(model_run_str_list):
    mr_list = []
    for d1 in model_run_str_list.keys():
        dtobj = dateutil.parser.parse(d1)
        dt_str = ('{dt.year}-{dt.month}-{dt.day} ' \
                  + '{dt.hour:02d}:{dt.minute:02d}').format(dt=dtobj)
        mr_list += [(dt_str, d1)]
    return mr_list


class ForestController(object):

    '''

    '''

    def __init__(self,
                 init_var,
                 init_time_ix,
                 datasets,
                 init_fcast_time,
                 plot_type_time_lookups,
                 plots,
                 bokeh_imgs,
                 colorbar_widget,
                 stats_widgets,
                 region_dict,
                 bokeh_doc,
                 feedback_dir,
                 bokeh_id,
                 ):

        '''
        
        '''

        self.data_time_slider = None
        self.model_var_dd = None
        self.time_prev_button = None
        self.time_next_button = None
        self.model_run_dd = None
        self.region_dd = None
        self.left_model_dd = None
        self.right_model_dd = None

        self.plots = plots
        self.bokeh_imgs = bokeh_imgs
        self.region_dict = region_dict
        self.plot_type_time_lookups = plot_type_time_lookups
        self.plot_names = list(self.plot_type_time_lookups.keys())
        self.datasets = datasets
        self.current_fcast_time = init_fcast_time
        self.current_var = init_var
        self.current_time_index = init_time_ix
        self._refresh_times(update_gui=False)

        self.init_time = self.available_times[self.current_time_index]
        self.num_times = self.available_times.shape[0]
        self.colorbar_div = colorbar_widget
        self.stats_widgets = stats_widgets
        self.bokeh_doc = bokeh_doc
        self.bokeh_id = bokeh_id
        self.feedback_dir = feedback_dir
        self.feedback_visible = False

        self.create_widgets()
        self.process_events = True


    def create_widgets(self):

        '''
        
        '''

        # Create variable selection dropdown widget
        variable_menu_list = \
            create_dropdown_opt_list_from_dict(VARIABLE_DD_DICT,
                                               self.plot_names)
        self.model_var_dd = \
            bokeh.models.widgets.Dropdown(label='Variable to visualise',
                                          menu=variable_menu_list,
                                          button_type='warning')
        self.model_var_dd.on_change('value', self.on_var_change)

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


        # select model run
        model_run_list = create_model_run_list(self.datasets)
        self.model_run_dd = \
            bokeh.models.widgets.Dropdown(label='Model run',
                                          menu=model_run_list,
                                          button_type='warning')
        self.model_run_dd.on_change('value', self._on_model_run_change)
        
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
                                                               self.datasets[self.current_fcast_time].keys())
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
        self.major_config_row = bokeh.layouts.row(self.model_var_dd, 
                                                  self.region_dd)

        config_path = os.path.join(CONFIG_DIR, FEEDBACK_CONF_FILENAME)

        self.feedback_gui = forest.feedback.UserFeedback(config_path,
                                                         self.feedback_dir,
                                                         self.bokeh_id,
                                                         )
        self.feedback_layout = self.feedback_gui.get_feedback_widgets()

        self.uf_vis_toggle = \
            bokeh.models.widgets.Toggle(label='Show feedback form',
                                        active=self.feedback_visible)
        self.uf_vis_toggle.on_click(self._on_uf_vis_toggle)
        self.uf_vis_layout = bokeh.layouts.column()

        self.time_row = \
            bokeh.layouts.row(self.time_prev_button,
                              bokeh.models.Spacer(width=20, height=60),
                              self.data_time_slider,
                              bokeh.models.Spacer(width=20, height=60),
                              self.time_next_button,
                              bokeh.models.Spacer(width=20, height=60),
                              self.model_run_dd,
                              )
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
                                                self.uf_vis_toggle,
                                                self.uf_vis_layout,
                                                )

    def on_time_prev(self):
        
        '''Event handler for changing to previous time step
        
        '''
        if not self.process_events:
            return
        print('selected previous time step')
        
        new_time = int(self.data_time_slider.value - 1)
        if new_time >= 0:
            self.current_time_index = new_time
            self.data_time_slider.value = self.current_time_index
        else:
            print('Cannot select time < 0')

    def on_time_next(self):
        
        '''
        
        '''
        
        if not self.process_events:
            return
        print('selected next time step')

        new_time = int(self.data_time_slider.value + 1)
        if new_time < self.num_times:
            self.current_time_index = new_time
            self.data_time_slider.value = self.current_time_index
        else:
            print('Cannot select time > num_times')      
        
    def on_data_time_change(self, attr1, old_val, new_val):

        '''Event handler for a change in the selected forecast data time.

        '''

        if not self.process_events:
            return
        print('data time handler')
        self.current_time_index = new_val

        for p1 in self.plots:
            new_time1 = self.available_times[self.current_time_index]
            # self.data_time_slider.label = 'Data time {0}'.format(new_time1)
            p1.set_data_time(new_time1)


    def _refresh_times(self, update_gui=True):
        self.available_times = \
            forest.data.get_available_times(
                self.datasets[self.current_fcast_time],
                self.plot_type_time_lookups[self.current_var])
        try:
            new_time = self.available_times[self.current_time_index]
        except IndexError:
            self.process_events = False
            self.current_time_index = 0
            new_time = self.available_times[self.current_time_index]
            if update_gui and self.data_time_slider:
                self.data_time_slider.value = self.current_time_index
            self.process_events = True

        if update_gui and self.data_time_slider:
            self.data_time_slider.end = self.available_times.shape[0]
        return new_time


    def on_var_change(self, attr1, old_val, new_val):

        '''Event handler for a change in the selected plot type.

        '''
        if not self.process_events:
            return

        print('var change handler')
        self.current_var = new_val
        new_time = self._refresh_times()
        for p1 in self.plots:
            # different variables have different times available, soneed to
            # set time when selecting a variable
            p1.current_time = new_time
            p1.set_var(new_val)

    def on_region_change(self, attr1, old_val, new_val):

        '''Event handler for a change in the selected plot region.

        '''
        if not self.process_events:
            return

        print('region change handler')

        for p1 in self.plots:
            p1.set_region(new_val)

    def on_config_change(self, plot_index, attr1, old_val, new_val):

        '''Event handler for a change in the selected model configuration output.

        '''

        if not self.process_events:
            return
        print('config change handler')
        
        self.plots[plot_index].set_config(new_val)

    def _on_model_run_change(self, attr1, old_val, new_val):
        if not self.process_events:
            return
        print('selected new model run {0}'.format(new_val))
        self.current_fcast_time = new_val
        new_time = self._refresh_times()
        for p1 in self.plots:
            # different variables have different times available, soneed to
            # set time when selecting a variable
            p1.current_time = new_time
            p1.set_dataset(self.datasets[self.current_fcast_time],
                           self.current_fcast_time)

    def _on_uf_vis_toggle(self, new1):
        if not self.process_events:
            return
        print('toggled feedback form visibility')
        self.feedback_visible = new1
        if self.feedback_visible:
            self.uf_vis_layout.children = [self.feedback_layout]
        else:
            self.uf_vis_layout.children = []

