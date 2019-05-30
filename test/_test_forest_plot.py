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


class TestForestPlotSetDataset(unittest.TestCase):
    def test_set_dataset(self):
        config = "config"
        old_forest_datasets = {
            config: "Old dataset"
        }
        plot_descriptions = {
            config: "Plot description"
        }
        plot_options = None
        plot_variable = "precipitation"
        figure_name = None
        region = "region"
        region_dict = {
            region: [0, 1, 0, 1]
        }
        app_path = "/some/path"
        initial_time = None
        forest_plot = forest.plot.ForestPlot(old_forest_datasets,
                                             plot_descriptions,
                                             plot_options,
                                             figure_name,
                                             plot_variable,
                                             config,
                                             region,
                                             region_dict,
                                             app_path,
                                             initial_time)
        new_forest_datasets = {
            config: "New dataset"
        }
        forest_plot.render = unittest.mock.Mock()
        forest_plot.set_dataset(new_forest_datasets)
        self.assertEqual(forest_plot.forest_datasets,
                         {"config": "New dataset"})


class TestForestPlotSetConfig(unittest.TestCase):
    def test_can_be_constructed(self):
        """the minimal information needed to construct a ForestPlot"""
        forest.plot.ForestPlot(*forest_plot_args())

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
        cube = iris.cube.Cube(self.data,
                              dim_coords_and_dims=[
                                  (latitude, 0),
                                  (longitude, 1)
                              ])
        return cube


class TestForestPlot(unittest.TestCase):
    def setUp(self):
        self.data = np.arange(4).reshape((2, 2))
        self.lons = [0, 1]
        self.lats = [0, 1]
        self.dataset = FakeDataset(self.data, self.lons, self.lats)

    def test_render_distinguishes_model_run_times(self):
        first_dataset = FakeDataset([[1, 2, 3], [4, 5, 6]], [0, 1, 2], [0, 1])
        second_dataset = FakeDataset([[4, 2, 3], [4, 5, 6]], [0, 1, 2], [0, 1])
        config = "config"
        datasets = {
            "config": first_dataset
        }
        options = {
            "precipitation": {
                "cmap": None,
                "norm": None
            }
        }
        descriptions = {
            "config": "Title"
        }
        variable = "precipitation"
        region = "region"
        regions = {
            "region": (0, 1, 0, 1)
        }
        plot = forest.plot.ForestPlot(datasets,
                                      descriptions,
                                      options,
                                      None,
                                      variable,
                                      config,
                                      region,
                                      regions,
                                      None,
                                      None)
        plot.render()
        before = plot.bokeh_img_ds.data["image"][0][:]
        plot.set_dataset({
            "config": second_dataset
        })
        plot.set_model_run_time("2018-01-01 12:00")
        plot.render()
        after = plot.bokeh_img_ds.data["image"][0][:]
        self.assertTrue(np.not_equal(before[:, :, :2], after[:, :, :2]).all())

    def test_render(self):
        plot = make_forest_plot(self.dataset)
        plot.render()

        # Assert bokeh ColumnDataSource correctly populated
        result = plot.bokeh_img_ds.data["image"][0]
        expect = [[[68, 1, 84, 255]]]
        np.testing.assert_array_equal(result, expect)

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


def make_forest_plot(dataset):
    """minimal data needed to call set_config"""
    datasets = {
        "current_config": dataset
    }
    plot_descriptions = {
        "current_config": "Label"
    }
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
    return forest.plot.ForestPlot(datasets,
                                  plot_descriptions,
                                  plot_options,
                                  figname,
                                  plot_var,
                                  conf1,
                                  reg1,
                                  rd1,
                                  app_path,
                                  init_time)


def forest_plot_args(plot_var='plot_var',
                     conf1='current_config',
                     plot_options=None,
                     init_time=None,
                     region_dict=None,
                     region="region"):
    """Helper to construct ForestPlot"""
    forest_datasets = {
        conf1: None
    }
    plot_descriptions = {
        conf1: None
    }
    figname = None
    if region_dict is None:
        region_dict = {
            region: [0, 1, 0, 1]
        }
    app_path = None
    init_time = None
    return (forest_datasets,
            plot_descriptions,
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


class TestVisability(unittest.TestCase):
    def setUp(self):
        self.data = np.arange(9).reshape((3, 3))
        self.lons = np.linspace(0, 10, 3)
        self.lats = np.linspace(0, 10, 3)
        self.dataset = FakeDataset(self.data, self.lons, self.lats)

    def test_when_visible_is_set_false_alpha_is_0(self):
        fplot = make_forest_plot(self.dataset)
        fplot.render()
        fplot.visible = False
        self.assertEqual(fplot.bokeh_image.glyph.global_alpha, 0)

    def test_when_visible_is_set_true_alpha_is_0(self):
        fplot = make_forest_plot(self.dataset)
        fplot.render()
        fplot.visible = True
        self.assertEqual(fplot.bokeh_image.glyph.global_alpha, 1)

    def test_bokeh_img_is_created_if_visible(self):
        fplot = make_forest_plot(self.dataset)
        fplot.visible = True
        self.assertIsNotNone(fplot.bokeh_image)
