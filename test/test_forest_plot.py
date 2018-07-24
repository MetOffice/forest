import unittest
import numpy as np
import iris
import forest.plot
import matplotlib


class FakeDataset(object):
    def get_data(self, var_name, selected_time):
        # Fake Cube generator
        nx, ny = 10, 10
        longitude = iris.coords.DimCoord(np.linspace(-180, 180, nx),
                                         standard_name="longitude",
                                         units="degrees")
        latitude = iris.coords.DimCoord(np.linspace(-90, 90, ny),
                                        standard_name="latitude",
                                        units="degrees")
        data = np.zeros((nx, ny), dtype=np.float)
        return iris.cube.Cube(data,
                              dim_coords_and_dims=[
                                  (longitude, 0),
                                  (latitude, 1)
                              ])


class TestForestPlot(unittest.TestCase):
    def test_create_plot(self):
        config = "config"
        dataset = {
            config: {
                "data": FakeDataset(),
                "data_type_name": None
            }
        }
        plot_options = {
            "mslp": {
                "cmap": None,
                "norm": None
            }
        }
        unit_dict_display = {
            "mslp": "display units"
        }
        model_run_time = "2018-01-01 00:00:00"
        args = self.args(plot_var="mslp",
                         conf1=config,
                         dataset=dataset,
                         plot_options=plot_options,
                         unit_dict_display=unit_dict_display,
                         model_run_time=model_run_time)
        forest_plot = forest.plot.ForestPlot(*args)
        forest_plot.create_plot()

    def test_create_plot_sets_main_plot(self):
        config = "config"
        dataset = {
            config: {
                "data": FakeDataset(),
                "data_type_name": None
            }
        }
        plot_options = {
            "mslp": {
                "cmap": None,
                "norm": None
            }
        }
        unit_dict_display = {
            "mslp": "display units"
        }
        model_run_time = "2018-01-01 00:00:00"
        args = self.args(plot_var="mslp",
                         conf1=config,
                         dataset=dataset,
                         plot_options=plot_options,
                         unit_dict_display=unit_dict_display,
                         model_run_time=model_run_time)
        forest_plot = forest.plot.ForestPlot(*args)
        forest_plot.create_plot()
        self.assertIsInstance(forest_plot.main_plot,
                              matplotlib.collections.QuadMesh)

    def test_plot_funcs_keys(self):
        forest_plot = forest.plot.ForestPlot(*self.args())
        result = sorted(forest_plot.plot_funcs.keys())
        expect = ['I',
                  'V',
                  'W',
                  'accum_precip_12hr',
                  'accum_precip_24hr',
                  'accum_precip_3hr',
                  'accum_precip_6hr',
                  'air_temperature',
                  'blank',
                  'cloud_fraction',
                  'himawari-8',
                  'mslp',
                  'precipitation',
                  'simim',
                  'wind_mslp',
                  'wind_streams',
                  'wind_vectors']
        self.assertEqual(result, expect)

    def args(self,
             plot_var='plot_var',
             conf1='current_config',
             dataset=None,
             plot_options=None,
             unit_dict_display=None,
             model_run_time=None):
        """Helper to construct ForestPlot"""
        if dataset is None:
            dataset = {
                conf1: {
                    'data_type_name': None
                }
            }
        figname = None
        reg1 = None
        rd1 = {
            reg1: [0, 1, 0, 1]
        }
        unit_dict = None
        app_path = None
        init_time = None
        return (dataset,
                model_run_time,
                plot_options,
                figname,
                plot_var,
                conf1,
                reg1,
                rd1,
                unit_dict,
                unit_dict_display,
                app_path,
                init_time)
