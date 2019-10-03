"""Redux design pattern"""
import copy
from functools import wraps
from forest.observe import Observable
from forest.export import export


__all__ = []


@export
def middleware(f):
    """Curries functions to satisfy middleware signature"""
    @wraps(f)
    def outer(*args):
        def inner(next_dispatch):
            def inner_most(action):
                f(*args, next_dispatch, action)
            return inner_most
        return inner
    return outer


@export
class Store(Observable):
    def __init__(self, reducer, initial_state=None, middlewares=None):
        self.reducer = reducer
        self.state = initial_state if initial_state is not None else {}
        if middlewares is not None:
            mws = [m(self) for m in middlewares]
            f = self.dispatch
            for mw in reversed(mws):
                f = mw(f)
            self.dispatch = f
        super().__init__()

    def dispatch(self, action):
        self.state = self.reducer(self.state, action)
        self.notify(self.state)
