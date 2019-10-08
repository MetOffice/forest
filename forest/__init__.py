"""
FOREST is a command line application built on top of
the Bokeh library that harnesses the power of scientific
software to visualise meteorological and oceanographic
forecasts alongside observations.

.. automodule:: forest.redux

.. automodule:: forest.keys

.. automodule:: forest.config

.. automodule:: forest.load

"""
__version__ = '0.4.3'

from .config import *
from . import (
        navigate,
        unified_model,
        redux,
        tutorial)
from .db import Database
from .keys import *
from .load import *
