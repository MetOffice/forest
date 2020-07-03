"""
FOREST is a command line application built on top of
the Bokeh library that harnesses the power of scientific
software to visualise meteorological and oceanographic
forecasts alongside observations.

.. automodule:: forest.redux

.. automodule:: forest.state
    :members:

.. automodule:: forest.rx

.. automodule:: forest.observe

.. automodule:: forest.series

.. automodule:: forest.colors

.. automodule:: forest.keys

.. automodule:: forest.config

.. automodule:: forest.geo

.. automodule:: forest.presets

.. automodule:: forest.services

"""
__version__ = '0.20.7'

from .config import *
from . import (
        actions,
        redux,
        tutorial)
from .db import Database
from .keys import *
from .reducer import reducer
