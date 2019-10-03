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


def test_key_press_middleware():
    middlewares = [forest.keys.navigate]
    store = forest.redux.Store(
            forest.db.reducer,
            middlewares=middlewares)
    store.dispatch(forest.keys.press("OSLeft"))
