import copy
from forest.observe import Observable


def reducer(state, action):
    state = copy.deepcopy(state)
    kind = action["kind"]
    if kind == "ACTION":
        state.update(action["payload"])
    return state


class Store(Observable):
    def __init__(self, reducer, middlewares=None):
        self.state = {}
        self.reducer = reducer
        if middlewares is None:
            middlewares = []
        self.middlewares = middlewares
        super().__init__()

    def dispatch(self, action):
        actions = self.pure(action)
        for middleware in self.middlewares:
            actions = self.bind(middleware, actions)
        for action in actions:
            self.state = self.reducer(self.state, action)

    @staticmethod
    def pure(action):
        yield action

    def bind(self, middleware, actions):
        store = self
        for action in actions:
            for _action in middleware(store, action):
                yield _action


class Log:
    def __init__(self):
        self.actions = []

    def __call__(self, store, action):
        self.actions.append(action)
        yield action


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
