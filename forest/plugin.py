"""Simple plugin architecture"""
import importlib


def call(entry_point):
    """Call entry_point to run plugin"""
    *parts, method = entry_point.split(".")
    module = importlib.import_module(".".join(parts))
    return getattr(module, method)()
