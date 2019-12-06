import pytest
import unittest.mock
from forest import dataset


@pytest.mark.parametrize("state,action,expect", [
    ({}, dataset.set_label("Label"), {"label": "Label"})
])
def test_reducer(state, action, expect):
    result = dataset.reducer(state, action)
    assert result == expect


def test_dataset_ui_emits_set_label():
    listener = unittest.mock.Mock()
    ui = dataset.DatasetUI()
    ui.subscribe(listener)
    ui.callback(None, None, "Label")
    listener.assert_called_once_with(dataset.set_label("Label"))
