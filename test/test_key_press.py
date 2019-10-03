import forest
import unittest.mock


def test_on_change_emits_action():
    code = 123
    listener = unittest.mock.Mock()
    key_press = forest.KeyPress()
    key_press.subscribe(listener)
    key_press.source.data = {'keys': [code]}
    action = forest.keys.press(code)
    listener.assert_called_once_with(action)


def test_navigate_maps_arrow_right_to_next_valid_time():
    check_key("ArrowRight", forest.db.next_value("valid_time", "valid_times"))


def test_navigate_maps_arrow_left_to_previous_valid_time():
    check_key("ArrowLeft", forest.db.previous_value("valid_time", "valid_times"))


def test_navigate_maps_arrow_up_to_next_initial_time():
    check_key("ArrowUp", forest.db.next_value("initial_time", "initial_times"))


def test_navigate_maps_arrow_down_to_previous_initial_time():
    check_key("ArrowDown", forest.db.previous_value("initial_time", "initial_times"))


def check_key(code, action):
    log = forest.db.Log()
    middlewares = [
            forest.keys.navigate,
            log]
    store = forest.redux.Store(
            forest.db.reducer,
            middlewares=middlewares)
    store.dispatch(forest.keys.press(code))
    actual = log.actions[0]
    expected = action
    assert actual == expected
