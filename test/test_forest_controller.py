import unittest
import bokeh.plotting
import forest.control


class FakeDataset(object):
    def __init__(self, times):
        self.times = times

    def get_times(self, variable):
        return self.times

    def get_data(self, variable, selected_time):
        return unittest.mock.Mock()


class FakePlot(object):
    bokeh_img_ds = None
    def set_data_time(self, time):
        pass


class TestForestController(unittest.TestCase):
    def test_on_time_next_given_two_times_moves_index_forward(self):
        initial_variable = None
        initial_time_index = 0
        initial_forecast_time = "2018-01-01 00:00:00"
        final_forecast_time = "2018-01-01 12:00:00"
        times = [initial_forecast_time,
                 final_forecast_time]
        datasets = {
            initial_forecast_time: {
                "key1": FakeDataset(times),
                "key2": FakeDataset(times)
            }
        }
        plot_type_time_lookups = {
            initial_variable: None
        }
        plots = [FakePlot(), FakePlot()]
        bokeh_figure = bokeh.plotting.figure()
        region_dict = {
            "region": None
        }
        controller = forest.control.ForestController(
            initial_variable,
            initial_time_index,
            datasets,
            initial_forecast_time,
            plot_type_time_lookups,
            plots,
            bokeh_figure,
            region_dict
        )
        controller.on_time_next()
        self.assertEqual(controller.num_times, 2)
        self.assertEqual(controller.current_time_index, 1)

    def test_on_model_run_change(self):
        initial_forecast_time = "2018-01-01 00:00:00"
        datasets = {
            initial_forecast_time: {
                "key": FakeDataset([initial_forecast_time])
            }
        }
        plot = self.make_forest_plot()
        plot.render = unittest.mock.Mock()
        controller = self.make_controller(initial_forecast_time,
                                          datasets,
                                          [plot, plot])
        attr, old, new = None, None, initial_forecast_time
        controller._on_model_run_change(attr, old, new)
        self.assertEqual(plot.forest_datasets, datasets[initial_forecast_time])

    def make_controller(self, initial_forecast_time, datasets, plots):
        initial_variable = None
        initial_time_index = 0
        plot_type_time_lookups = {
            initial_variable: None
        }
        bokeh_figure = bokeh.plotting.figure()
        region_dict = {
            "region": None
        }
        controller = forest.control.ForestController(
            initial_variable,
            initial_time_index,
            datasets,
            initial_forecast_time,
            plot_type_time_lookups,
            plots,
            bokeh_figure,
            region_dict
        )
        return controller

    def make_forest_plot(self):
        initial_time = None
        forest_datasets = {"config": FakeDataset([initial_time])}
        plot_variable = "precipitation"
        plot_descriptions = None
        plot_options = {"precipitation": {"cmap": None, "norm": None}}
        figure_name = None
        config = "config"
        region = "region"
        regions = {"region": [0, 1, 0, 1]}
        app_path = None
        return forest.plot.ForestPlot(forest_datasets,
                                      plot_descriptions,
                                      plot_options,
                                      figure_name,
                                      plot_variable,
                                      config,
                                      region,
                                      regions,
                                      app_path,
                                      initial_time)
