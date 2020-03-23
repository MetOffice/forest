"""
FOREST is a command line application built on top of
the Bokeh library that harnesses the power of scientific
software to visualise meteorological and oceanographic
forecasts alongside observations.

.. automodule:: forest.redux

.. automodule:: forest.rx

.. automodule:: forest.observe

.. automodule:: forest.series

.. automodule:: forest.colors

.. automodule:: forest.keys

.. automodule:: forest.config

.. automodule:: forest.load

.. automodule:: forest.geo

.. automodule:: forest.presets

"""
__version__ = '0.13.1'

from .config import *
from . import (
        navigate,
        redux,
        tutorial)
from .db import Database
from .keys import *
from .load import *
