import pytest
import unittest
import os
import netCDF4
import numpy as np
import numpy.testing as npt
import datetime as dt
import bokeh.plotting
from forest import position, redux, rx

def test_series_set_position_action():
    action = position.set_position(0, 0)
    assert action == {"kind": position.SET_POSITION, "payload": {"x": 0, "y": 0}}


def test_tap_listener_on_tap_emits_action():
    x, y = 1, 2  # different values to assert order
    listener = unittest.mock.Mock()
    figure = bokeh.plotting.figure()
    event = bokeh.events.Tap(figure, x=x, y=y)
    tap_listener = position.TapListener()
    tap_listener.add_subscriber(listener)
    tap_listener.update_xy(event)
    listener.assert_called_once_with(position.set_position(x, y))


def test_position_draw_mark():
    figure = bokeh.plotting.figure()
    marker = position.MarkDraw(figure)
    pos = {"x": 0, "y": 0}
    marker.place_marker(pos)

