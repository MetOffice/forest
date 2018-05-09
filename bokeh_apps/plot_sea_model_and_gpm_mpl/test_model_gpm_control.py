import unittest
import model_gpm_control


class TestModelGpmControl(unittest.TestCase):
    def test_can_be_constructed(self):
        init_var = ""
        datasets = []
        init_time_ix = 0
        init_fcast_time = ""
        plot_list = []
        bokeh_img_list = []
        stats_list = []
        model_gpm_control.ModelGpmControl(init_var,
                                          datasets,
                                          init_time_ix,
                                          init_fcast_time,
                                          plot_list,
                                          bokeh_img_list,
                                          stats_list)
