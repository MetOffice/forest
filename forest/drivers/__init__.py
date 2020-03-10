from importlib import import_module
from forest.exceptions import DriverNotFound


def get_dataset(driver_name, settings=None):
    """Find Dataset related to file type"""
    if settings is None:
        settings = {}
    try:
        module = import_module(f"forest.drivers.{driver_name}")
    except ModuleNotFoundError:
        raise DriverNotFound(driver_name)
    return module.Dataset(**settings)
