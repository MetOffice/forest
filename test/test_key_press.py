import pytest
import forest
import unittest.mock
from forest.db import next_value, previous_value


def test_on_change_emits_action():
    code = 123
    listener = unittest.mock.Mock()
    key_press = forest.KeyPress()
    key_press.add_subscriber(listener)
    key_press.source.data = {'keys': [code]}
    action = forest.keys.press(code)
    listener.assert_called_once_with(action)


@pytest.mark.parametrize("code,action", [
    ("ArrowRight", next_value("valid_time", "valid_times")),
    ("ArrowLeft", previous_value("valid_time", "valid_times")),
    ("ArrowUp", next_value("initial_time", "initial_times")),
    ("ArrowDown", previous_value("initial_time", "initial_times")),
])
def test_key(code, action):
    store = forest.redux.Store(forest.db.reducer)
    actual = list(forest.keys.navigate(store, forest.keys.press(code)))
    expected = [action]
    assert actual == expected
