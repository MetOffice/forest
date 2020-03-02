import pytest
import unittest
import os
import netCDF4
import numpy as np
import numpy.testing as npt
import datetime as dt
import bokeh.plotting
from forest import screen, redux, rx, db

def test_position_reducer():
    state = screen.reducer({}, screen.set_position(0, 0))
    assert state == {"position": {"x": 0, "y": 0}}

def test_position_reducer_immutable_state():
    state = {"position": {"x": 1, "y": 1}}
    next_state = screen.reducer(state, screen.set_position(0, 0))
    assert state == {"position": {"x": 1, "y": 1}}
    assert next_state == {"position": {"x": 0, "y": 0}}

@pytest.mark.parametrize("actions,expect", [
    ([], {}),
    ([screen.set_position(0, 0)], {"position": {"x": 0, "y": 0}}),
    ([db.set_value("key", "value")], {"key": "value"}),
    ([
        screen.set_position(0, 0),
        db.set_value("key", "value")], {
            "key": "value",
            "position": {"x": 0, "y": 0}}),
])
def test_combine_reducers(actions, expect):
    reducer = redux.combine_reducers(screen.reducer, db.reducer)
    state = {}
    for action in actions:
        state = reducer(state, action)
    assert state == expect

def test_series_set_position_action():
    action = screen.set_position(0, 0)
    assert action == {"kind": screen.SET_POSITION, "payload": {"x": 0, "y": 0}}

def test_tap_listener_on_tap_emits_action():
    x, y = 1, 2  # different values to assert order
    listener = unittest.mock.Mock()
    figure = bokeh.plotting.figure()
    event = bokeh.events.Tap(figure, x=x, y=y)
    tap_listener = screen.TapListener()
    tap_listener.add_subscriber(listener)
    tap_listener.update_xy(event)
    listener.assert_called_once_with(screen.set_position(x, y))


def test_position_draw_mark():
    figure = bokeh.plotting.figure()
    marker = screen.MarkDraw(figure)
    pos = {"x": 0, "y": 0}
    marker.place_marker(pos)

