import unittest.mock
import pytest
from forest import colors, main
import bokeh.models
import numpy as np


@pytest.mark.parametrize("state,action,expect", [
    ({}, colors.fixed_on(), {"colorbar": {"fixed": True}}),
    ({}, colors.fixed_off(), {"colorbar": {"fixed": False}}),
])
def test_reducer(state, action, expect):
    result = colors.reducer(state, action)
    assert expect == result


def test_color_controls():
    color_mapper = bokeh.models.LinearColorMapper()
    controls = colors.Controls(color_mapper, "Accent", 3)
    controls.render()
    assert color_mapper.palette == ['#7fc97f', '#beaed4', '#fdc086']


def test_controls_on_name():
    color_mapper = bokeh.models.LinearColorMapper()
    controls = colors.Controls(color_mapper, "Accent", 3)
    controls.on_name(None, None, "Blues")
    assert color_mapper.palette == bokeh.palettes.all_palettes["Blues"][3]
    # assert controls.names.label == "Blues"
    # assert controls.numbers.label == "3"
    assert controls.dropdowns["numbers"].menu == [(str(i), str(i))
            for i in [3, 4, 5, 6, 7, 8, 9]]


def test_controls_on_number():
    color_mapper = bokeh.models.LinearColorMapper()
    controls = colors.Controls(color_mapper, "Accent", 3)
    controls.on_number(None, None, 5)
    assert color_mapper.palette == bokeh.palettes.all_palettes["Accent"][5]


def test_controls_on_reverse():
    attr, old, new = None, [], [0]
    color_mapper = bokeh.models.LinearColorMapper()
    name = "Accent"
    number = 3
    controls = colors.Controls(color_mapper, name, number)
    controls.on_reverse(attr, old, new)
    assert color_mapper.palette == ['#fdc086', '#beaed4', '#7fc97f']


def test_mapper_limits():
    attr, old, new = None, None, None  # not used by callback
    color_mapper = bokeh.models.LinearColorMapper()
    source = bokeh.models.ColumnDataSource({
        "image": [
            np.arange(9).reshape(3, 3)
        ]
    })
    mapper_limits = colors.MapperLimits([source], color_mapper)
    mapper_limits.on_source_change(attr, old, new)
    assert color_mapper.low == 0
    assert color_mapper.high == 8


@pytest.mark.parametrize("new,action", [
    ([0], colors.fixed_on()),
    ([], colors.fixed_off()),
])
def test_mapper_limits_on_fixed(new, action):
    color_mapper = bokeh.models.LinearColorMapper()
    sources = [bokeh.models.ColumnDataSource({
        "image": [np.arange(9).reshape(3, 3)]
    })]
    mapper_limits = colors.MapperLimits(sources, color_mapper)
    listener = unittest.mock.Mock()
    mapper_limits.subscribe(listener)
    mapper_limits.on_checkbox_change(None, None, new)
    listener.assert_called_once_with(action)
