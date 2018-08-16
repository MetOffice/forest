import unittest
import bokeh.plotting
import forest.control


class FakeDataset(object):
    def __init__(self, times):
        self.times = times

    def get_times(self, variable):
        return self.times


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
                "key1": {
                    "data": FakeDataset(times)
                },
                "key2": {
                    "data": FakeDataset(times)
                }
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
