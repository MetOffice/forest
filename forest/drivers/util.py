import importlib


def by_name(driver_name):
    """Select appropriate driver by name

    Drivers are implemented as modules inside forest.drivers

    :returns: module that implements driver interface
    """
    return importlib.import_module(f"forest.drivers.{driver_name}")
