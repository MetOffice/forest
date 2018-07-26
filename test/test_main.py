import unittest
import unittest.mock
import os
import sys
TEST_DIR = os.path.dirname(os.path.realpath(__file__))
APP_DIR = os.path.join(TEST_DIR, "../bokeh_apps/plot_sea_model_and_gpm_mpl/")
sys.path.insert(0, APP_DIR)
# import main


@unittest.skip("cherry pick")
class TestMain(unittest.TestCase):
    def setUp(self):
        self.init_fcast_time = "20180101"
        self.gpm_dataset = "GpmDataset"

    @unittest.mock.patch("main.forest.plot.ForestPlot")
    @unittest.mock.patch("main.forest.data.get_available_times")
    @unittest.mock.patch("main.forest.data.get_available_datasets")
    @unittest.mock.patch("main.model_gpm_data")
    @unittest.mock.patch("main.model_gpm_control")
    def test_main_calls_modelgpmcontrol(self,
                                        model_gpm_control,
                                        model_gpm_data,
                                        get_available_datasets,
                                        get_available_times,
                                        ForestPlot):
        model_gpm_data.GpmDataset.return_value = self.gpm_dataset
        get_available_datasets.return_value = self.init_fcast_time, {self.init_fcast_time: {}}
        get_available_times.return_value = [None, None, None, None, None]
        with unittest.mock.patch("main.bokeh") as bokeh:
            main.main(bokeh_id="")
            args, kwargs = model_gpm_control.ModelGpmControl.call_args
            expect = {
               self.init_fcast_time: {
                   "gpm_imerg_early": {
                       "data": self.gpm_dataset,
                       "data_type_name": "GPM IMERG Early",
                       "gpm_type": "early"
                   },
                   "gpm_imerg_late": {
                       "data": self.gpm_dataset,
                       "data_type_name": "GPM IMERG Late",
                       "gpm_type": "late"
                   },
               }
            }
            self.assertEqual(args[1], expect)
