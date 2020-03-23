from importlib import import_module
from forest.exceptions import DriverNotFound
from functools import wraps


_CACHE = {}


def _cache(f):
    # Ensure per-server dataset instances
    def wrapped(driver_name, settings=None):
        uid = _uid(driver_name, settings)
        if uid not in _CACHE:
            _CACHE[uid] = f(driver_name, settings)
        return _CACHE[uid]
    return wrapped


def _uid(driver_name, settings):
    if settings is None:
        return (driver_name,)
    return (driver_name,) + tuple(_maybe_hashable(settings[k])
                                  for k in sorted(settings.keys()))


def _maybe_hashable(value):
    if isinstance(value, list):
        return tuple(value)
    return value


@_cache
def get_dataset(driver_name, settings=None):
    """Find Dataset related to file type"""
    if settings is None:
        settings = {}
    try:
        module = import_module(f"forest.drivers.{driver_name}")
    except ModuleNotFoundError:
        raise DriverNotFound(driver_name)
    return module.Dataset(**settings)
