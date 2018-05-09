import unittest
import model_gpm_control


class TestModelGpmControl(unittest.TestCase):
    def setUp(self):
        self.init_fcast_time = "20180101"

    def test_can_be_constructed(self):
        init_var = ""
        mock_gpm_dataset = unittest.mock.Mock()
        mock_gpm_dataset.get_times.return_value = [
            self.init_fcast_time
        ]
        datasets = {self.init_fcast_time: {
            "gpm_imerg_early": {
                "data": mock_gpm_dataset
            }
        }}
        init_time_ix = 0
        init_fcast_time = self.init_fcast_time
        plot_list = []
        bokeh_img_list = []
        stats_list = [None, None]
        with unittest.mock.patch("model_gpm_control.bokeh") as bokeh:
            model_gpm_control.ModelGpmControl(init_var,
                                              datasets,
                                              init_time_ix,
                                              init_fcast_time,
                                              plot_list,
                                              bokeh_img_list,
                                              stats_list)
