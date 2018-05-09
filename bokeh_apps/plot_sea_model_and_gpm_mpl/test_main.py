import unittest
import unittest.mock
import main


class TestMain(unittest.TestCase):
    def setUp(self):
        self.init_fcast_time = "20180101"

    def test_main_can_be_called(self):
        with unittest.mock.patch("main.bokeh") as bokeh:
            main.main(bokeh_id="")

    @unittest.mock.patch("main.forest.plot.ForestPlot")
    @unittest.mock.patch("main.forest.data.get_available_times")
    @unittest.mock.patch("main.forest.data.get_available_datasets")
    @unittest.mock.patch("main.model_gpm_control")
    def test_main_calls_modelgpmcontrol(self,
                                        model_gpm_control,
                                        get_available_datasets,
                                        get_available_times,
                                        ForestPlot):
        get_available_datasets.return_value = self.init_fcast_time, {self.init_fcast_time: {}}
        get_available_times.return_value = [None, None, None, None, None]
        with unittest.mock.patch("main.bokeh") as bokeh:
            main.main(bokeh_id="")
            model_gpm_control.ModelGpmControl.assert_called_once_with()
