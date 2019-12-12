import unittest.mock
import copy
from forest.redux import Store
from forest.observe import Observable


def reducer(state, action):
    state = copy.deepcopy(state)
    kind = action["kind"]
    if kind == "ACTION":
        state.update(action["payload"])
    return state


def duplicate(store, action):
    yield action
    yield action


def action():
    return {"kind": "ACTION", "payload": {"key": "value"}}


def test_reducer():
    store = Store(reducer)
    store.dispatch(action())
    assert store.state == {"key": "value"}


def test_middleware():
    log = unittest.mock.Mock(return_value=())
    store = Store(reducer, middlewares=[duplicate, log])
    store.dispatch(action())
    log.assert_has_calls([
        unittest.mock.call(store, action()),
        unittest.mock.call(store, action()),
    ])
