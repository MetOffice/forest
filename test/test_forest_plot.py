import unittest
<<<<<<< HEAD
import unittest.mock
import forest


class TestForestPlot(unittest.TestCase):
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

    @unittest.mock.patch("forest.plot.matplotlib")
    @unittest.mock.patch("forest.plot.bokeh")
    def test_forest_plot_calls_bokeh_plotting_figure(self,
                                                     bokeh,
                                                     matplotlib):
        """ForestPlot constructs its own bokeh.Figure instance

        .. note:: Every time create_plot() is called a call
                  to create_bokeh_img_plot_from_fig() inside
                  which bokeh.plotting.figure() is called

        .. warn:: ForestPlot does not initialise a reference
                  to self.current_figure that is done by
                  self.create_matplotlib_fig()

        .. note:: ForestPlot.create_matplotlib_fig() expects
                  self.current_var to be a key in self.plot_funcs

        .. note:: forest.util.get_image_array_from_figure() takes
                  a matplotlib.Figure instance
        """
        figure = matplotlib.pyplot.figure.return_value
        with unittest.mock.patch("forest.util") as forest_util:
            fixture = forest.plot.ForestPlot(*self.generic_args())
            fixture.plot_funcs["plot_variable"] = unittest.mock.Mock()
            fixture.create_matplotlib_fig()
            fixture.create_bokeh_img_plot_from_fig()
            forest_util.get_image_array_from_figure.assert_called_once_with(figure)
        tools = "pan,wheel_zoom,reset,save,box_zoom"
        x_range = bokeh.models.Range1d.return_value
        y_range = bokeh.models.Range1d.return_value
        bokeh.plotting.figure.assert_called_once_with(plot_height=600,
                                                      plot_width=800,
                                                      tools=tools,
                                                      x_range=x_range,
                                                      y_range=y_range)

    def test_forest_plot_should_accept_bokeh_figure(self):
        """To open forest to allow generic layouts

        Decoupling the design to allow app specific bokeh figures
        will make managing layouts easier
        """
        fake_figure = "bokeh figure"
        forest_plot = forest.plot.ForestPlot(*self.generic_args(),
                                             bokeh_figure=fake_figure)
        self.assertEqual(forest_plot.bokeh_figure, fake_figure)

    @unittest.skip("developing toggle controller")
    @unittest.mock.patch("forest.util")
    @unittest.mock.patch("forest.plot.matplotlib")
    @unittest.mock.patch("forest.plot.bokeh")
    def test_forest_plot_should_keep_bokeh_figure(self,
                                                  bokeh,
                                                  matplotlib,
                                                  util):
        """A user supplied figure should be retained throughout object
        life cycle"""
        fake_figure = "bokeh figure"
        fake_plot_func = unittest.mock.Mock()
        forest_plot = forest.plot.ForestPlot(*self.generic_args(),
                                             bokeh_figure=fake_figure)
        forest_plot.plot_funcs[forest_plot.current_var] = fake_plot_func
        forest_plot.create_plot()
        self.assertEqual(forest_plot.bokeh_figure, fake_figure)

    def test_get_data_returns_cube(self):
        """Separation of concerns needed to separate Forest infrastructure from plotting"""
        class FakeData(object):
            def get_data(self, var_name, selected_time):
                pass
        fake_data = FakeData()
        dataset, *remaining_args = self.generic_args()
        dataset = {
            "current_config": {
                "data_type_name": None,
                "data": fake_data
            }
        }
        forest_plot = forest.plot.ForestPlot(dataset, *remaining_args)
        forest_plot.get_data()

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
=======
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
>>>>>>> 7cd6bc6... added unit test suite to understand ForestPlot design
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
