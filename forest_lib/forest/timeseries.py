"""Forest time series"""


class ForestTimeSeries():
    """
    Class representing a time series plot. Unlike a map based plot, where
    each plot represents a single dataset, the timeseries plot plot several
    datasets together in the same plot for comparison purposes.
    """

    def __init__(self,
                 datasets,
                 model_run_time,
                 selected_point,
                 current_var):
        """

        :param datasets: A dictionary object of datasets
        :param model_run_time: A string representing the model run to be
                               displayed. All configs for the given model run
                               will be displayed on the timeseries graph.
        :param selected_point: The lat/long coordinates of the point to
                               display the timeseries for.
        :param current_var: The current variable to be displyed
                            e.g. precipitation
        """
        self.datasets = datasets
        self.model_run_time = model_run_time
        self.current_point = selected_point
        self.current_fig = None
        self.current_var = current_var
        self.cds_dict = {}

        self.placeholder_data = {'x_values': [0.0, 1.0],
                                 'y_values': [0.0, 0.0]}

    def __str__(self):
        """

        :return: A string describing the class.
        """
        return 'Class representing a time series plot in the forest tool'

    def create_plot(self):
        """
        Create the timeseries plot in a bokeh figure. This is where the actual
        work is done.
        :return: A bokeh figure object containing the timeseries plot.
        """
        self.current_fig = bokeh.plotting.figure(tools=BOKEH_TOOLS_LIST)
        self.bokeh_lines = {}
        self.cds_list = {}
        for ds_name in self.datasets.keys():
            current_ds = self.datasets[ds_name]['data']
            times1 = current_ds.get_times(self.current_var)
            times1 = times1 - times1[0]
            var_cube = \
                current_ds.get_timeseries(self.current_var,
                                          self.current_point)
            if var_cube:
                var_values = var_cube.data

                data1 = {'x_values': times1,
                         'y_values': var_values}

                ds_source = bokeh.models.ColumnDataSource(data=data1)

                ds_line_plot = self.current_fig.line(x='x_values',
                                                     y='y_values',
                                                     source=ds_source,
                                                     name=ds_name)
            else:
                ds_source = \
                    bokeh.models.ColumnDataSource(data=self.placeholder_data)
                ds_line_plot = self.current_fig.line(x='x_values',
                                                     y='y_values',
                                                     source=ds_source,
                                                     name=ds_name)

            self.cds_dict[ds_name] = ds_source
            self.bokeh_lines[ds_name] = ds_line_plot

        self._update_fig_title()

        return self.current_fig

    def _update_plot(self):
        """
        Update the bokeh figure with a new timeseries plot. This is called by
        functions that are called to change an input, and should not be called
        by a user directly.
        :return:
        """
        for ds_name in self.datasets.keys():
            if self.cds_dict[ds_name] is not None:
                current_ds = self.datasets[ds_name]['data']
                times1 = current_ds.get_times(self.current_var)
                times1 = times1 - times1[0]
                var_cube = \
                    current_ds.get_timeseries(self.current_var,
                                              self.current_point)

                if var_cube is not None:
                    var_values = var_cube.data

                    data1 = {'x_values': times1,
                             'y_values': var_values}

                    self.cds_dict[ds_name].data = data1
                else:
                    self.cds_dict[ds_name].data = self.placeholder_data
        self._update_fig_title()

    def _update_fig_title(self):
        """
        Update the title of the bokeh figure.
        :return: No return.
        """
        fig_title = 'Plotting variable {var} for model run {mr}'
        fig_title = fig_title.format(var=self.current_var,
                                     mr=str(self.model_run_time))
        self.current_fig.title.text = fig_title

    def set_var(self, new_var):
        """
        Set the timeseries to display the
        :param new_var: The new variable to be displayed.
        :return: No return value
        """
        self.current_var = new_var
        self._update_plot()

    def set_selected_point(self, latitude, longitude):
        """
        Set a new location for the timeseries display.
        :param latitude: Latitude of the location to display.
        :param longitude: longitude of the location to display.
        :return: No return value.
        """
        self.current_point = (latitude, longitude)
        self._update_plot()

    def set_data_time(self, new_time):
        """
        This function is provided for a consistent interface with other plot
        classes, but does nothing.
        """
        pass

    def set_dataset(self, new_dataset, new_model_run_time):
        """
        Set a new model run as the input to the timeseries.
        :param new_dataset: The dictionary representing the new model run
                            dataset.
        :param new_model_run_time: A string representing the model run time.
        :return: No return value.
        """
        self.datasets = new_dataset
        self.model_run_time = new_model_run_time
