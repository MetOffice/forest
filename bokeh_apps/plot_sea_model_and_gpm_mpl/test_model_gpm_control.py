import unittest
import forest
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

    def test_gpm_imerg_radio_button_group_constructed_with_arguments(self, bokeh):
        self.make_model_gpm_controller(self.datasets)
        self.assert_any_call(bokeh.models.widgets.RadioButtonGroup,
                             unittest.mock.call(labels=["GPM IMERG Early", "GPM IMERG Late"],
                                                button_type='warning',
                                                width=800,
                                                active=0),
                             "GPM IMERG RadioButtonGroup not created/created incorrectly")

    def assert_any_call(self, mocked_object, expected_call, message):
        """custom assertion to extend default unittest.mock.assert_any_call message"""
        try:
            if len(expected_call) == 2:
                args, kwargs = expected_call
            elif len(expected_call) == 3:
                _, args, kwargs = expected_call
            mocked_object.assert_any_call(*args, **kwargs)
        except AssertionError as error:
            lines = [message]
            lines.append("actual calls:")
            for call in mocked_object.call_args_list:
                lines.append("\t {}".format(str(call)))
            lines.append("expected call:")
            lines.append("\t{}".format(expected_call))
            error.args = ("\n".join(lines),)
            raise error

    def test_imerg_labels_given_realistic_dictionary_returns_labels(self, bokeh):
        fixture = self.make_model_gpm_controller(self.datasets)
        expect = ["GPM IMERG Early", "GPM IMERG Late"]
        self.assertEqual(fixture.imerg_labels, expect)

    def test_imerg_labels_given_duplicate_labels_returns_unique_labels(self, bokeh):
        class FakeDataset(object):
            def __init__(self, times):
                self.times = times
            def get_times(self, variable):
                return self.times
        fixture = self.make_model_gpm_controller({
            "20180101": {
                "gpm_imerg_early": {
                    "data": FakeDataset(["20180101"]),
                    "data_type_name": "GPM IMERG Early"
                }
            },
            "20180102": {
                "gpm_imerg_early": {
                    "data": FakeDataset(["20180102"]),
                    "data_type_name": "GPM IMERG Early"
                }
            }
        })
        expect = ["GPM IMERG Early"]
        self.assertEqual(fixture.imerg_labels, expect)

    def test_imerg_labels_given_minimal_dictionary_returns_empty_list(self, bokeh):
        class FakeDataset(object):
            def get_times(self, variable):
                return [None]
        fake_dataset = FakeDataset()
        fixture = self.make_model_gpm_controller({
            self.init_fcast_time: {
                "key": {
                    "data": fake_dataset
                }
            }
        })
        self.assertEqual(fixture.imerg_labels, [])

    def test_on_imerg_change(self, bokeh):
        """simulate what happens when an IMERG button is clicked

        .. note: the callback used is a curried version of the
                 method with the first argument set to 1
        """
        plot_index = 1
        attr1 = None
        old_val = 0
        new_val = 1
        mock_plot = unittest.mock.Mock()
        plot_list = [None, mock_plot]
        controller = self.make_model_gpm_controller(self.datasets,
                                                    plot_list=plot_list)
        controller.on_imerg_change(plot_index,
                                   attr1,
                                   old_val,
                                   new_val)
        mock_plot.set_config.assert_called_once_with("gpm_imerg_late")

    def make_model_gpm_controller(self, datasets,
                                  plot_list=()):
        init_var = ""
        init_time_ix = 0
        init_fcast_time = self.init_fcast_time
        bokeh_img_list = []
        stats_list = [None, None]
        return model_gpm_control.ModelGpmControl(init_var,
                                                 datasets,
                                                 init_time_ix,
                                                 init_fcast_time,
                                                 plot_list,
                                                 bokeh_img_list,
                                                 stats_list)
