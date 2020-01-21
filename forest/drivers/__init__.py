import importlib


def load_driver(name):
    """Load a driver by name"""
    return importlib.import_module(f"forest.drivers.{name}")
