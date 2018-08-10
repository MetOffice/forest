import unittest
import unittest.mock
import sys
import io
import numpy as np
import iris
import forest
import forest.plot
import matplotlib
import matplotlib.pyplot as plt
import bokeh.plotting


class TestForestPlotSetRegion(unittest.TestCase):
    """Forest callback to set_region should only change bokeh extents"""
    def setUp(self):
        new_region = "new region"
        old_region = "old region"
        self.x_start, self.x_end, self.y_start, self.y_end = 1, 2, 3, 4
        region_dict = {
            new_region: [
                self.y_start,
                self.y_end,
                self.x_start,
                self.x_end
            ],
            old_region: [1, 2, 1, 2]
        }
        self.bokeh_figure = bokeh.plotting.figure()
        self.forest_plot = forest.plot.ForestPlot(*forest_plot_args(
            region=old_region,
            region_dict=region_dict
        ), bokeh_figure=self.bokeh_figure)
        self.forest_plot.set_region(new_region)

    def test_set_region_sets_bokeh_figure_x_start(self):
        self.assertEqual(self.bokeh_figure.x_range.start, self.x_start)

    def test_set_region_sets_bokeh_figure_x_end(self):
        self.assertEqual(self.bokeh_figure.x_range.end, self.x_end)

    def test_set_region_sets_bokeh_figure_y_start(self):
        self.assertEqual(self.bokeh_figure.y_range.start, self.y_start)

    def test_set_region_sets_bokeh_figure_y_end(self):
        self.assertEqual(self.bokeh_figure.y_range.end, self.y_end)



class TestForestPlotSetConfig(unittest.TestCase):
    def test_can_be_constructed(self):
        """the minimal information needed to construct a ForestPlot"""
        forest.plot.ForestPlot(*forest_plot_args())

    @unittest.mock.patch("forest.plot.bokeh")
    def test_set_config(self, bokeh):
        """minimal data needed to call set_config"""
        data = np.arange(9).reshape((3, 3))
        longitudes = np.linspace(0, 10, 3)
        latitudes = np.linspace(0, 10, 3)
        fake_dataset = FakeDataset(data=data,
                                   longitudes=longitudes,
                                   latitudes=latitudes)
        dataset = {
            "current_config": {
                "data_type_name": None
            },
            "new_config": {
                "data_type_name": "Label",
                "data": fake_dataset
            }
        }
        model_run_time = None
        plot_options = {
            "precipitation": {
                "cmap": None,
                "norm": None
            }
        }
        figname = None
        plot_var = "precipitation"
        conf1 = "current_config"
        reg1 = "current_region"
        rd1 = {
            "current_region": (0, 1, 0, 1)
        }
        app_path = None
        init_time = None
        forest_plot = forest.plot.ForestPlot(dataset,
                                             model_run_time,
                                             plot_options,
                                             figname,
                                             plot_var,
                                             conf1,
                                             reg1,
                                             rd1,
                                             app_path,
                                             init_time)
        # System under test
        forest_plot.set_config("new_config")

        # Assertions
        self.assertEqual(forest_plot.current_config, "new_config")
        self.assertEqual(forest_plot.plot_description, "Label")

    def test_forest_plot_should_accept_bokeh_figure(self):
        """To open forest to allow generic layouts

        Decoupling the design to allow app specific bokeh figures
        will make managing layouts easier
        """
        fake_figure = unittest.mock.Mock()
        forest_plot = forest.plot.ForestPlot(*self.generic_args(),
                                             bokeh_figure=fake_figure)
        self.assertEqual(forest_plot.bokeh_figure, fake_figure)

    def generic_args(self, plot_var="plot_variable"):
        """Helper to construct ForestPlot"""
        return forest_plot_args(plot_var=plot_var)


class TestForestPlotPressureLevelsHPa(unittest.TestCase):
    def test_pressure_levels_hpa(self):
        args = forest_plot_args()
        forest_plot = forest.plot.ForestPlot(*args)
        result = list(forest_plot.PRESSURE_LEVELS_HPA)
        expect = [
              980,  982,  984,  986,  988,  990,
              992,  994,  996,  998, 1000, 1002,
             1004, 1006, 1008, 1010, 1012, 1014,
             1016, 1018, 1020, 1022, 1024, 1026,
             1028
        ]
        self.assertEqual(result, expect)


class FakeDataset(object):
    def __init__(self, data, longitudes, latitudes):
        self.data = data
        self.longitudes = longitudes
        self.latitudes = latitudes

    def get_data(self, var_name, selected_time):
        # Fake Cube generator
        longitude = iris.coords.DimCoord(self.longitudes,
                                         standard_name="longitude",
                                         units="degrees")
        latitude = iris.coords.DimCoord(self.latitudes,
                                        standard_name="latitude",
                                        units="degrees")
        return iris.cube.Cube(self.data,
                              dim_coords_and_dims=[
                                  (longitude, 0),
                                  (latitude, 1)
                              ])


class TestForestPlot(unittest.TestCase):
    """
    .. note:: test suite swallows stdout and stderr to prevent
              confusion when running other tests
    """
    def test_create_plot(self):
        fake_data = np.arange(4).reshape((2, 2))
        fake_lons = [0, 1]
        fake_lats = [0, 1]
        config = "config"
        dataset = {
            config: {
                "data": FakeDataset(fake_data,
                                    fake_lons,
                                    fake_lats),
                "data_type_name": None
            }
        }
        plot_var = 'mslp'
        plot_options = {
            "mslp": {
                "cmap": None,
                "norm": None
            }
        }
        model_run_time = "2018-01-01 00:00:00"
        init_time=None
        figure_name = None
        region="region"
        region_dict = {
            region: [0, 1, 0, 1]
        }
        app_path = None
        forest_plot = forest.plot.ForestPlot(
            dataset,
            model_run_time,
            plot_options,
            figure_name,
            plot_var,
            config,
            region,
            region_dict,
            app_path,
            init_time
        )
        forest_plot.create_plot()

        # Assert bokeh ColumnDataSource correctly populated
        result = forest_plot.bokeh_img_ds.data["image"][0]
        expect = [[[68, 1, 84, 255]]]
        np.testing.assert_array_equal(result, expect)

    @unittest.skip("understanding other test")
    def test_create_plot_sets_main_plot(self):
        config = "config"
        fake_data = np.zeros((2, 2))
        dataset = {
            config: {
                "data": FakeDataset(fake_data),
                "data_type_name": None
            }
        }
        plot_options = {
            "mslp": {
                "cmap": None,
                "norm": None
            }
        }
        model_run_time = "2018-01-01 00:00:00"
        args = self.args(plot_var="mslp",
                         conf1=config,
                         dataset=dataset,
                         plot_options=plot_options,
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

    def args(self, *args, **kwargs):
        return forest_plot_args(*args, **kwargs)


def forest_plot_args(plot_var='plot_var',
                     conf1='current_config',
                     dataset=None,
                     plot_options=None,
                     init_time=None,
                     model_run_time=None,
                     region_dict=None,
                     region="region"):
    """Helper to construct ForestPlot"""
    if dataset is None:
        dataset = {
            conf1: {
                'data_type_name': None
            }
        }
    figname = None
    if region_dict is None:
        region_dict = {
            region: [0, 1, 0, 1]
        }
    app_path = None
    init_time = None
    return (dataset,
            model_run_time,
            plot_options,
            figname,
            plot_var,
            conf1,
            region,
            region_dict,
            app_path,
            init_time)


class TestPColorMesh(unittest.TestCase):
    def test_get_array(self):
        fake_data = np.arange(4).reshape((2, 2))
        result = plt.pcolormesh(fake_data).get_array()
        expect = [0, 1, 2, 3]
        np.testing.assert_array_equal(result, expect)

    def test_to_rgba_given_bytes_true(self):
        x = np.array([0, 1])
        y = np.array([0, 1])
        z = np.arange(4).reshape((2, 2))
        quad_mesh = plt.pcolormesh(x, y, z)
        result = quad_mesh.to_rgba(quad_mesh.get_array(),
                                   bytes=True)
        expect = [[68, 1, 84, 255]]
        np.testing.assert_array_equal(result, expect)


import skimage.transform
class TestSmoothImage(unittest.TestCase):
    def test_skimage_transform_resize(self):
        image = np.ones((10, 10, 4))
        output_shape = (5, 5)
        result = skimage.transform.resize(image,
                                          output_shape)
        result = result.astype(np.uint8)
        expect = np.ones((5, 5, 4), dtype=np.uint8)
        np.testing.assert_array_equal(expect, result)
