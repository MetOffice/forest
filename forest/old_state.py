"""Decorator to map dict to namedtuple state"""
from functools import wraps
import forest.db.control


def old_state(f):
    @wraps(f)
    def wrapper(*args):
        if len(args) == 2:
            self, state = args
            return f(self, _to_old(state))
        else:
            (state,) = args
            return f(_to_old(state))

    return wrapper


def _to_old(state):
    kwargs = {k: state.get(k, None) for k in forest.db.control.State._fields}
    return forest.db.control.State(**kwargs)


def unique(f):
    previous = None
    called = False

    @wraps(f)
    def wrapper(*args):
        nonlocal previous
        nonlocal called

        if len(args) == 2:
            self, value = args
            key = (id(self), value)  # Distinguish wrapped methods
        else:
            (value,) = args
            key = value

        if (not called) or (key != previous):
            called = True
            previous = key
            if len(args) == 2:
                result = f(self, value)
            else:
                result = f(value)
            return result

    return wrapper
