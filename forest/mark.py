"""Decorators to mark classes and functions"""
import inspect
from unittest.mock import Mock
from contextlib import contextmanager
from functools import wraps
from forest.observe import Observable
import pandas as pd
import numpy as np


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


def sql_sanitize_time(*labels):
    """Decorator to protect SQL statements from unsupported datetime types

    >>> @sql_sanitize_time("b", "c")
    ... def method(self, a, b, c=None, d=False):
    ...     # b and c will be converted to a str compatible with SQL queries
    ...     pass

    """
    def outer(f):
        parameters = inspect.signature(f).parameters

        # Get positional index
        index = {}
        for i, name in enumerate(parameters):
            if name in labels:
                index[name] = i

        def inner(*args, **kwargs):
            args = list(args)
            for label in labels:
                if label in kwargs:
                    kwargs[label] = sanitize_time(kwargs[label])
                else:
                    i = index[label]
                    if i < len(args):
                        args[i] = sanitize_time(args[i])
            return f(*args, **kwargs)
        return inner
    return outer


def sanitize_time(value):
    """Query-compatible equivalent of value"""
    fmt = "%Y-%m-%d %H:%M:%S"
    if value is None:
        return value
    elif isinstance(value, str):
        return value
    elif isinstance(value, np.datetime64):
        return pd.to_datetime(str(value)).strftime(fmt)
    else:
        return value.strftime(fmt)
