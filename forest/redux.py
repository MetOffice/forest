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

.. autofunction:: combine_reducers

"""
import copy
from functools import wraps
from forest.observe import Observable
from forest.export import export
from typing import Callable, Iterable


# Type aliases
Action = dict
State = dict
Reducer = Callable[[State, Action], State]
Middleware = Callable[['Store', Action], Iterable[Action]]


__all__ = []


@export
def combine_reducers(*reducers):
    """Simple combine passes action and state to all reducers

    :returns: reducer function
    """
    def wrapped(state, action):
        state = copy.deepcopy(state)
        for reducer in reducers:
            state = reducer(state, action)
        return state
    return wrapped


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

    >>> store.add_subscriber(listener)

    :param reducer: function combines action and state to produce new state
    :param initial_state: optional initial state, default {}
    :param middlewares: list of middleware functions that intercept actions
    """
    def __init__(self, reducer, initial_state=None, middlewares=()):
        self.reducer = reducer
        self.state = initial_state if initial_state is not None else {}
        self.middlewares = middlewares
        self.in_progress = False
        self.queue = []
        super().__init__()

    def dispatch(self, action):
        """Apply reducer and notify listeners of new state

        :param action: plain dict consumed by the reducer
        """
        if self.in_progress:
            # Add asynchronous action to backlog
            self.queue.append(action)
            return

        # Synchronous processing
        self.in_progress = True
        self.sync_process(action)

        # Process backlog
        actions = list(self.queue)
        self.queue = []
        for action in actions:
            self.sync_process(action)

        self.in_progress = False

    def sync_process(self, action):
        """Pass action through middleware/reducer pipeline"""
        actions = self.pure(action)
        for middleware in self.middlewares:
            actions = self.bind(middleware, self, actions)
        for _action in actions:
            self.state = self.reducer(self.state, _action)
            self.notify(self.state)

    @staticmethod
    def pure(action):
        """Embed action into action generator"""
        yield action

    @staticmethod
    def bind(middleware, store, actions):
        """Flat map action generators from middleware into action generator"""
        for action in actions:
            yield from middleware(store, action)
