import unittest
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
            self.assertEqual(forest_plot.plot_description, "Label")
