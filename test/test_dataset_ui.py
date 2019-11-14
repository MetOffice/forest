import unittest.mock
from forest.ui import DatasetUI
from forest.db.control import set_value


def test_dataset_ui_emits_set_label():
    listener = unittest.mock.Mock()
    ui = DatasetUI()
    ui.subscribe(listener)
    ui.callback(None, None, "Label")
    listener.assert_called_once_with(set_value("label", "Label"))
