"""
Redux design pattern
--------------------

As applications grow the number of components
vying to keep their state synchronised becomes unwieldy.
A better way to manage state is centralise it
so that there is a single source of truth. The
consequence of which is easy to replay, serialise
and rehydrate applications.

For further reading, check out the redux.js docs

* https://redux.js.org/introduction/motivation

It also enables an easily unit testable system
since state updates are pure function with easy to
separate single responsibility components. Middleware
components are plug and play, drop one in to add
behaviour to your app without fear of altering
existing behaviour.

.. autoclass:: Store
   :members:

.. autofunction:: middleware

"""
import copy
from functools import wraps
from forest.observe import Observable
from forest.export import export


__all__ = []


@export
def middleware(f):
    """Curries functions to satisfy middleware signature

    This decorator supports both instance methods and functions
    signatures. It nests functions into a format used by
    :class:`Store` to patch its dispatch method.

    * Method signature `method(self, store, next_dispatch, action)`
    * Function signature `func(store, next_dispatch, action)`

    :param: function to decorate should have middleware signature
    """
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
    """Observable state container

    The redux design pattern is a simple way to
    keep track of state changes. A reducer combines an action with
    the current state to produce the next state. The reducer
    should be a pure function in the sense of not having side effects.

    Non-pure behaviour can be incorporated through the use of
    middleware. Middleware takes an action and either passes it on,
    filters it, enriches it or emits new actions.

    The store is an observable that emits states, views can register
    themselves with the store to receive the latest states as and
    when they are created.

    >>> store.subscribe(listener)

    :param reducer: function combines action and state to produce new state
    :param initial_state: optional initial state, default {}
    :param middlewares: list of middleware functions that intercept actions
    """
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
        """Apply reducer and notify listeners of new state

        :param action: plain dict consumed by the reducer
        """
        self.state = self.reducer(self.state, action)
        self.notify(self.state)
