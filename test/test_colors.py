import unittest.mock
from forest import colors, main
import bokeh.models
import numpy as np


def test_color_controls():
    color_mapper = bokeh.models.LinearColorMapper()
    name = "Accent"
    number = 3
    controls = colors.Controls(color_mapper, name, number)
    controls.render()
    assert color_mapper.palette == ['#7fc97f', '#beaed4', '#fdc086']


def test_colors_on_reverse():
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


def test_mapper_limits_on_checkbox_change_emit_fixed():
    attr, old, new = None, [], [0]
    color_mapper = bokeh.models.LinearColorMapper()
    sources = [bokeh.models.ColumnDataSource({
        "image": [
            np.arange(9).reshape(3, 3)
        ]
    })]
    mapper_limits = colors.MapperLimits(sources, color_mapper)
    listener = unittest.mock.Mock()
    mapper_limits.subscribe(listener)
    mapper_limits.on_checkbox_change(attr, old, new)
    listener.assert_called_once_with(colors.fixed_on())
