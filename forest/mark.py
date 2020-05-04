"""Decorators to mark classes and functions"""
from unittest.mock import Mock
from contextlib import contextmanager
from functools import wraps
from forest.observe import Observable


def component(cls):
    """Enforce one-way data-flow"""
    if issubclass(cls, Observable) and hasattr(cls, "render"):
        cls.render = disable_notify(cls.render)
    return cls


def disable_notify(render):
    """Disable self.notify during self.render"""
    @wraps(render)
    def wrapper(self, *args, **kwargs):
        with disable(self, "notify"):
            return_value = render(self, *args, **kwargs)
        return return_value
    return wrapper


@contextmanager
def disable(obj, method_name):
    """Temporarily disable a method inside a code block"""
    method = getattr(obj, method_name)
    setattr(obj, method_name, Mock())
    yield
    setattr(obj, method_name, method)
