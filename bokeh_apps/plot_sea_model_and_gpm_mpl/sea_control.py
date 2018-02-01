import functools

import bokeh
import bokeh.model
import bokeh.layouts

class SEA_control(object):
    def __init__(self,
                 datasets,
                 init_time,
                 num_times,
                 plot_list,
                 bokeh_img_list,
                 ):
        self.datasets =datasets
        self.init_time = init_time
        self.num_times = num_times
        self.plot_list = plot_list
        self.bokeh_img_list = bokeh_img_list
        self.create_widgets()

    def __str__(self):
        return 'MVC-style controller class for model rainfall vs GPM app.'

    def create_widgets(self):
        # Set up bokeh widgets
        def create_dropdown_opt_list(iterable1):
            return [(k1, k1) for k1 in iterable1]


        self.data_time_slider = bokeh.models.widgets.Slider(start=0,
                                                       end=self.num_times,
                                                       value=self.init_time,
                                                       step=3,
                                                       title="Data time",
                                                       width=800)

        self.data_time_slider.on_change('value', self.on_data_time_change)

        self.model_menu_list = create_dropdown_opt_list([ds_name for ds_name in self.datasets.keys()
                                                    if 'imerg' not in ds_name])
        self.gpm_imerg_menu_list = create_dropdown_opt_list([ds_name for ds_name in self.datasets.keys()
                                                        if 'imerg' in ds_name])

        model_dd_desc = 'Model display'
        self.model_dd = bokeh.models.widgets.Dropdown(menu=self.model_menu_list,
                                                 label=model_dd_desc,
                                                 button_type='warning',
                                                 width=800)
        self.model_dd.on_change('value', functools.partial(self.on_config_change, 0))

        self.imerg_rbg = bokeh.models.widgets.RadioButtonGroup(labels=[ds_name for ds_name
                                                                  in self.datasets.keys()
                                                                  if 'imerg' in ds_name],
                                                          button_type='warning',
                                                          width=800)
        self.imerg_rbg.on_change('active', functools.partial(self.on_imerg_change, 1))

        # Set layout for widgets
        self.slider_row = bokeh.layouts.row(self.data_time_slider)
        self.config_row = bokeh.layouts.row(self.model_dd, self.imerg_rbg, width=1600)

        self.plots_row = bokeh.layouts.row(*self.bokeh_img_list)

        self.main_layout = bokeh.layouts.column(self.slider_row,
                                                self.config_row,
                                                self.plots_row,
                                                )

    def on_data_time_change(self, attr1, old_val, new_val):
        '''
        Event handler for a change in the selected forecast data time.
        '''
        print('selected new time {0}'.format(new_val))
        current_time = int(new_val / 3)
        for p1 in self.plot_list:
            p1.set_data_time(current_time)

    def on_date_slider_change(self, attr1, old_val, new_val):
        '''
        Event handler for a change in the selected forecast data date.
        '''
        print('selected new date {0}'.format(new_val))
        current_time = new_val.strftime('%Y%m%d') + self.current_time[-4:]
        for p1 in self.plot_list:
            p1.set_data_time(current_time)

    def on_hour_slider_change(self, attr1, old_val, new_val):
        '''
        Event handler for a change in the selected forecast data date.
        '''
        print('selected new date {0}'.format(new_val))
        current_time = self.current_time[:-4] + '{:02d}00'.format(new_val)
        for p1 in self.plot_list:
            p1.set_data_time(current_time)

    def on_config_change(self, plot_index, attr1, old_val, new_val):
        '''
        Event handler for a change in the selected model configuration output.
        '''
        print('selected new config {0}'.format(new_val))
        self.plot_list[plot_index].set_config(new_val)

    def on_imerg_change(self, plot_index, attr1, old_val, new_val):
        '''
        Event handler for a change in the selected model configuration output.
        '''
        imerg_list = [ds_name for ds_name in self.datasets.keys()
                      if 'imerg' in ds_name]
        print('selected new config {0}'.format(imerg_list[new_val]))
        new_config = imerg_list[new_val]
        self.plot_list[plot_index].set_config(new_config)
