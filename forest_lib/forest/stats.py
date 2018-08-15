"""Forest statistics widget"""
import bokeh.models
import forest.data


class ForestStats(object):
    """Compute and display summary statistics"""
    def __init__(self, unit_dict, unit_dict_display):
        self._plot_keys = ['precipitation',
                           'accum_precip_3hr',
                           'accum_precip_6hr',
                           'accum_precip_12hr',
                           'accum_precip_24hr',
                           'wind_vectors',
                           'wind_mslp',
                           'wind_streams',
                           'mslp',
                           'air_temperature',
                           'cloud_fraction',
                           'himawari-8',
                           'simim',
                           'W',
                           'I',
                           'V',
                           'blank']
        self.selected_point = None
        self.unit_dict = unit_dict
        self.unit_dict_display = unit_dict_display
        self.stats_widget = None
        self.stats_string = ''
        self.stats_data_var = dict([(k1,k1) for k1 in self._plot_keys])
        self.stats_data_var['wind_vectors'] = forest.data.WIND_SPEED_NAME
        self.stats_data_var['wind_mslp'] = forest.data.WIND_SPEED_NAME
        self.stats_data_var['wind_streams'] = forest.data.WIND_SPEED_NAME

    def create_widget(self):
        self.stats_widget = bokeh.models.widgets.Div(
            text=self.stats_string,
            height=200,
            width=400,
        )
        return self.stats_widget

    def update_widget(self):
        print('Updating stats widget')
        try:
            self.stats_widget.text = self.stats_string
        except AttributeError as e1:
            print('Unable to update stats as stats widget not initiated')

    def set_selected_point(self, latitude, longitude):
        self.selected_point = (latitude, longitude)
        if self.stats_widget:
            current_data = \
                self.get_data(self.stats_data_var[self.current_var])
            self.update(current_data)
            self.update_widget()

    def get_data(self, variable):
        raise NotImplementedError("Get data not implemented yet")

    def update(self, current_cube):
        data_to_process = current_cube.data
        stats_str_list = [self.current_title]
        unit_str = self.unit_dict_display[self.current_var]
        max_val = numpy.max(data_to_process)
        min_val = numpy.min(data_to_process)
        mean_val = numpy.mean(data_to_process)
        std_val = numpy.std(data_to_process)
        rms_val = numpy.sqrt(numpy.mean(numpy.power(data_to_process, 2.0)))
        model_run_info = 'Current model run start time: '
        mr_dtobj = dateutil.parser.parse(self.model_run_time)
        model_run_info += '{dt.year:d}-{dt.month:02d}-{dt.day:02d} '
        model_run_info += '{dt.hour:02d}{dt.minute:02d}Z'
        model_run_info = model_run_info.format(dt=mr_dtobj)

        selected_pt_info = 'No point selected'
        if self.selected_point is not None:
            if self.selected_point[0] > 0.0:
                lat_str = '{0:.2f} N'.format(abs(self.selected_point[0]))
            else:
                lat_str = '{0:.2f} S'.format(abs(self.selected_point[0]))

            if self.selected_point[1] > 0.0:
                long_str = '{0:.2f} E'.format(abs(self.selected_point[1]))
            else:
                long_str = '{0:.2f} W'.format(abs(self.selected_point[1]))


            sample_pts = [('latitude', self.selected_point[0]),
                          ('longitude', self.selected_point[1])]
            select_val_cube = \
                current_cube.interpolate(sample_pts,
                                            iris.analysis.Linear())

            select_val = float(select_val_cube.data)

            field_val_str = \
                'value: {val:.2f} {unit_str}'.format(val=select_val,
                                                     unit_str=unit_str)

            selected_pt_info = 'selected point {lat},{long}<br>'
            selected_pt_info += 'field value {fv}'
            selected_pt_info = selected_pt_info.format(lat=lat_str,
                                                       long=long_str,
                                                       fv=field_val_str)

        stats_str_list += [model_run_info,'']
        stats_str_list += [selected_pt_info, '']
        stats_str_list += ['Max = {0:.4f} {1}'.format(max_val, unit_str)]
        stats_str_list += ['Min = {0:.4f} {1}'.format(min_val, unit_str)]
        stats_str_list += ['Mean = {0:.4f} {1}'.format(mean_val, unit_str)]
        stats_str_list += ['STD = {0:.4f} {1}'.format(std_val, unit_str)]
        stats_str_list += ['RMS = {0:.4f} {1}'.format(rms_val, unit_str)]
        self.stats_string = '</br>'.join(stats_str_list)
