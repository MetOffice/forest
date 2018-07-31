import unittest
import unittest.mock
import sys
import io
import numpy as np
import iris
import forest
import forest.plot
import matplotlib
import bokeh.plotting
from collections import namedtuple
import array


class TestCoastlines(unittest.TestCase):
    @unittest.mock.patch("forest.plot.cartopy")
    def test_coastlines(self, cartopy):
        figure = bokeh.plotting.figure()
        forest.plot.coastlines(figure)
        cartopy.feature.COASTLINE.geometries.assert_called_once_with()


Extent = namedtuple("Extent", ["x_start", "x_end",
                               "y_start", "y_end"])


class TestClipXY(unittest.TestCase):
    def setUp(self):
        self.x = np.array([1, 2, 3])
        self.y = np.array([4, 5, 6])

    def test_clip_xy(self):
        """helper to restrict coastlines to bokeh figure"""
        extent = Extent(-np.inf, np.inf, -np.inf, np.inf)
        expect = self.x, self.y
        self.check_clip_xy(self.x, self.y, extent, expect)

    def test_clip_xy_removes_points_left_of_extent(self):
        extent = Extent(1.5, np.inf, -np.inf, np.inf)
        expect = self.x[1:], self.y[1:]
        self.check_clip_xy(self.x, self.y, extent, expect)

    def test_clip_xy_removes_points_right_of_extent(self):
        extent = Extent(-np.inf, 2.5, -np.inf, np.inf)
        expect = self.x[:-1], self.y[:-1]
        self.check_clip_xy(self.x, self.y, extent, expect)

    def test_clip_xy_removes_points_below_extent(self):
        extent = Extent(-np.inf, np.inf, 4.5, np.inf)
        expect = self.x[1:], self.y[1:]
        self.check_clip_xy(self.x, self.y, extent, expect)

    def test_clip_xy_removes_points_above_extent(self):
        extent = Extent(-np.inf, np.inf, -np.inf, 5.5)
        expect = self.x[:-1], self.y[:-1]
        self.check_clip_xy(self.x, self.y, extent, expect)

    def test_clip_xy_supports_array_array(self):
        """should handle array.array returned by cartopy geometries"""
        x = array.array("f", [1, 2, 3])
        y = array.array("f", [4, 5, 6])
        extent = Extent(-np.inf, np.inf, -np.inf, 5.5)
        expect = x[:-1], y[:-1]
        self.check_clip_xy(x, y, extent, expect)

    def check_clip_xy(self, x, y, extent, expect):
        result = forest.plot.clip_xy(x, y, extent)
        np.testing.assert_array_equal(result, expect)


class TestForestPlotSetConfig(unittest.TestCase):
    def test_can_be_constructed(self):
        """the minimal information needed to construct a ForestPlot"""
        dataset = {
            "current_config": {
                "data_type_name": None
            }
        }
        model_run_time = None
        po1 = None
        figname = None
        plot_var = "plot_variable"
        conf1 = "current_config"
        reg1 = "current_region"
        rd1 = {
            "current_region": None
        }
        unit_dict = None
        unit_dict_display = None
        app_path = None
        init_time = None
        forest.plot.ForestPlot(dataset,
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

    def test_set_config(self):
        """minimal data needed to call set_config"""
        dataset = {
            "current_config": {
                "data_type_name": None
            },
            "new_config": {
                "data_type_name": "Label"
            }
        }
        model_run_time = None
        po1 = None
        figname = None
        plot_var = "plot_variable"
        conf1 = "current_config"
        reg1 = "current_region"
        rd1 = {
            "current_region": None
        }
        unit_dict = None
        unit_dict_display = None
        app_path = None
        init_time = None
        mock_plot = unittest.mock.Mock()
        with unittest.mock.patch("forest.plot.matplotlib") as matplotlib:
            forest_plot = forest.plot.ForestPlot(dataset,
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
            forest_plot.plot_funcs[plot_var] = mock_plot
            forest_plot.update_bokeh_img_plot_from_fig = unittest.mock.Mock()
            # System under test
            forest_plot.set_config("new_config")

            # Assertions
            forest_plot.update_bokeh_img_plot_from_fig.assert_called_once_with()
            mock_plot.assert_called_once_with()
            self.assertEqual(forest_plot.current_config, "new_config")
            self.assertEqual(forest_plot.plot_description, "Label")

    def test_forest_plot_should_accept_bokeh_figure(self):
        """To open forest to allow generic layouts

        Decoupling the design to allow app specific bokeh figures
        will make managing layouts easier
        """
        fake_figure = "bokeh figure"
        forest_plot = forest.plot.ForestPlot(*self.generic_args(),
                                             bokeh_figure=fake_figure)
        self.assertEqual(forest_plot.bokeh_figure, fake_figure)

    def generic_args(self, plot_var="plot_variable"):
        """Helper to construct ForestPlot"""
        dataset = {
            "current_config": {
                "data_type_name": None
            }
        }
        model_run_time = None
        po1 = None
        figname = None
        conf1 = "current_config"
        reg1 = "current_region"
        rd1 = {
            "current_region": [0, 1, 0, 1]
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


@unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
@unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
class TestForestPlot(unittest.TestCase):
    """
    .. note:: test suite swallows stdout and stderr to prevent
              confusion when running other tests
    """
    def test_create_plot(self, stdout, stderr):
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
        # WARNING: no assertions made

    def test_create_plot_sets_main_plot(self, stdout, stderr):
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

    def test_plot_funcs_keys(self, stdout, stderr):
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
             unit_dict=None,
             unit_dict_display=None,
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
        unit_dict = None
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
                unit_dict,
                unit_dict_display,
                app_path,
                init_time)
