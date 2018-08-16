import unittest
import bokeh.plotting
import forest.control


class FakeDataset(object):
    def get_times(self, variable):
        return []


class FakePlot(object):
    bokeh_img_ds = None


class TestForestController(unittest.TestCase):
    def test_init_method(self):
        initial_variable = None
        initial_time_index = None
        initial_forecast_time = "2018-01-01 00:00:00"
        datasets = {
            initial_forecast_time: {
                "key1": {
                    "data": FakeDataset()
                },
                "key2": {
                    "data": FakeDataset()
                }
            }
        }
        plot_type_time_lookups = {
            initial_variable: None
        }
        plots = [FakePlot(), FakePlot()]
        bokeh_figure = bokeh.plotting.figure()
        colorbar_widget = None
        region_dict = {
            "region": None
        }
        feedback_dir = None
        bokeh_id = None
        forest.control.ForestController(
            initial_variable,
            initial_time_index,
            datasets,
            initial_forecast_time,
            plot_type_time_lookups,
            plots,
            bokeh_figure,
            colorbar_widget,
            region_dict,
            feedback_dir,
            bokeh_id
        )
