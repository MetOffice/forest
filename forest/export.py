"""Helper to export names in modules"""
import sys


def export(obj):
    __all__ = sys.modules[obj.__module__].__all__
    if obj.__name__ not in __all__:
        __all__.append(obj.__name__)
    return obj
