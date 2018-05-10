import unittest
import model_gpm_control


@unittest.mock.patch("model_gpm_control.bokeh")
class TestModelGpmControl(unittest.TestCase):
    def setUp(self):
        self.init_fcast_time = "20180101"
        mock_gpm_dataset = unittest.mock.Mock()
        mock_gpm_dataset.get_times.return_value = [
            self.init_fcast_time
        ]
        self.datasets = {
            self.init_fcast_time: {
                "gpm_imerg_early": {
                    "data": mock_gpm_dataset,
                    "data_type_name": "GPM IMERG Early",
                    "gpm_type": "early"
                },
                "gpm_imerg_late": {
                    "data": mock_gpm_dataset,
                    "data_type_name": "GPM IMERG Late",
                    "gpm_type": "late"
                }
            }
        }

    def test_radio_button_group_constructed_with_imerg_early_late(self, bokeh):
        self.make_radio_button_group(self.datasets)
        bokeh.models.widgets.RadioButtonGroup.assert_any_call(
            labels=[],
            button_type='warning',
            width=800
        )

    def test_imerg_labels_given_realistic_dictionary(self, bokeh):
        fixture = self.make_radio_button_group(self.datasets)
        self.assertEqual(fixture.imerg_labels(), [])

    def test_imerg_labels_given_minimal_dictionary_returns_empty_list(self, bokeh):
        class FakeDataset(object):
            def get_times(self, variable):
                return [None]
        fake_dataset = FakeDataset()
        fixture = self.make_radio_button_group({
            self.init_fcast_time: {
                "key": {
                    "data": fake_dataset
                }
            }
        })
        self.assertEqual(fixture.imerg_labels(), [])

    def make_radio_button_group(self, datasets):
        init_var = ""
        init_time_ix = 0
        init_fcast_time = self.init_fcast_time
        plot_list = []
        bokeh_img_list = []
        stats_list = [None, None]
        return model_gpm_control.ModelGpmControl(init_var,
                                                 datasets,
                                                 init_time_ix,
                                                 init_fcast_time,
                                                 plot_list,
                                                 bokeh_img_list,
                                                 stats_list)
