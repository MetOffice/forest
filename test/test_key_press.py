import pytest
import forest
import unittest.mock
from forest.db import (
        next_initial_time,
        previous_initial_time,
        next_valid_time,
        previous_valid_time)
from forest.components.time import toggle_animation


def test_on_change_emits_action():
    code = 123
    listener = unittest.mock.Mock()
    key_press = forest.KeyPress()
    key_press.add_subscriber(listener)
    key_press.source.data = {'keys': [code]}
    action = forest.keys.press(code)
    listener.assert_called_once_with(action)


@pytest.mark.parametrize("code,action", [
    ("ArrowRight", next_valid_time()),
    ("ArrowLeft", previous_valid_time()),
    ("ArrowUp", next_initial_time()),
    ("ArrowDown", previous_initial_time()),
    ("Space", toggle_animation()),
])
def test_key(code, action):
    store = forest.redux.Store(forest.db.reducer)
    actual = list(forest.keys.navigate(store, forest.keys.press(code)))
    expected = [action]
    assert actual == expected
