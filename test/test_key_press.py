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
    log = forest.db.Log()
    middlewares = [
            forest.keys.navigate,
            log]
    store = forest.redux.Store(
            forest.db.reducer,
            middlewares=middlewares)
    store.dispatch(forest.keys.press("ArrowRight"))
    actual = log.actions[0]
    expected = forest.db.next_value("valid_time", "valid_times")
    assert actual == expected


def test_navigate_maps_arrow_left_to_previous_valid_time():
    log = forest.db.Log()
    middlewares = [
            forest.keys.navigate,
            log]
    store = forest.redux.Store(
            forest.db.reducer,
            middlewares=middlewares)
    store.dispatch(forest.keys.press("ArrowLeft"))
    actual = log.actions[0]
    expected = forest.db.previous_value("valid_time", "valid_times")
    assert actual == expected
