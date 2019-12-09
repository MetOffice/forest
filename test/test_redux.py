import copy
from forest.redux import Store
from forest.middlewares import Log
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
    log = Log()
    store = Store(reducer, middlewares=[duplicate, log])
    store.dispatch(action())
    assert log.actions == [action(), action()]
