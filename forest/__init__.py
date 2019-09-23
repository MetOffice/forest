"""
This version of FOREST consists of a main.py program served
by Bokeh. Later releases will focus on dividing the API into re-usable
components.
"""
__version__ = '0.3.0'

from .config import *
from . import (
        navigate,
        unified_model)
