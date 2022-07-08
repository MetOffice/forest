from importlib import import_module
from forest.exceptions import DriverNotFound
from functools import wraps


def iter_drivers():
    """Generate builtin drivers"""
    import os

    directory = os.path.dirname(os.path.realpath(__file__))
    for path in os.listdir(directory):
        if path.endswith(".py") and not path.startswith("_"):
            # TODO introspect code to test for Dataset class
            yield path[:-3]


_CACHE = {}


def _cache(f):
    # Ensure per-server dataset instances
    def wrapped(driver_name, settings=None):
        uid = _uid(driver_name, settings)
        import json

        uid = json.dumps(uid)
        if uid not in _CACHE:
            _CACHE[uid] = f(driver_name, settings)
        return _CACHE[uid]

    return wrapped


def _uid(driver_name, settings):
    if settings is None:
        return (driver_name,)
    return (driver_name,) + tuple(
        _maybe_hashable(settings[k]) for k in sorted(settings.keys())
    )


def _maybe_hashable(value):
    if isinstance(value, list):
        return tuple(value)
    return value


@_cache
def get_dataset(driver_name, settings=None):
    """Find Dataset related to file type"""
    if settings is None:
        settings = {}
    return get_driver(driver_name)(**settings)


def get_driver(driver_name):
    """Get a top-level class related to a driver name

    .. note: TODO clean up ambiguity between driver and dataset in this context
    """
    try:
        # Try builtin driver first
        module = import_module(f"forest.drivers.{driver_name}")
    except ModuleNotFoundError:
        try:
            # Try user-defined driver second
            module = import_module(driver_name)
        except ModuleNotFoundError:
            raise DriverNotFound(driver_name)
    return module.Dataset
