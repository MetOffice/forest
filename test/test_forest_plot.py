import unittest
import numpy as np
import iris
import forest.plot


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
        args = self.args(plot_var="mslp",
                         conf1=config,
                         dataset=dataset)
        forest_plot = forest.plot.ForestPlot(*args)
        forest_plot.create_plot()

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
             dataset=None):
        """Helper to construct ForestPlot"""
        if dataset is None:
            dataset = {
                conf1: {
                    'data_type_name': None
                }
            }
        model_run_time = None
        po1 = None
        figname = None
        reg1 = None
        rd1 = {
            reg1: [None, None, None, None]
        }
        unit_dict = None
        unit_dict_display = None
        app_path = None
        init_time = None
        return (dataset,
                model_run_time,
                po1,
                figname,
                plot_var,
                conf1,
                reg1,
                rd1,
                unit_dict,
                unit_dict_display,
                app_path,
                init_time)
