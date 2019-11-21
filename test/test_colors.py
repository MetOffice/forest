import unittest.mock
import pytest
from forest import colors, main, redux
import bokeh.models
import numpy as np


@pytest.mark.parametrize("state,action,expect", [
    ({}, colors.fixed_on(), {"colorbar": {"fixed": True}}),
    ({}, colors.fixed_off(), {"colorbar": {"fixed": False}}),
    ({}, colors.set_palette_name("Accent"), {"colorbar": {"name": "Accent"}}),
    ({}, colors.set_palette_number(3), {"colorbar": {"number": 3}}),
    ({}, colors.set_palette_numbers([1, 2 ,3]), {"colorbar": {"numbers": [1, 2, 3]}}),
])
def test_reducer(state, action, expect):
    result = colors.reducer(state, action)
    assert expect == result


def test_color_controls():
    color_mapper = bokeh.models.LinearColorMapper()
    controls = colors.Controls(color_mapper)
    controls.render({"colorbar": {"name": "Accent", "number": 3}})
    assert color_mapper.palette == ['#7fc97f', '#beaed4', '#fdc086']


def test_controls_on_name():
    listener = unittest.mock.Mock()
    color_mapper = bokeh.models.LinearColorMapper()
    controls = colors.Controls(color_mapper)
    controls.subscribe(listener)
    controls.on_number(None, None, 5)
    listener.assert_called_once_with(colors.set_palette_number(5))


class Log:
    def __init__(self):
        self.actions = []

    @redux.middleware
    def __call__(self, store, next_dispatch, action):
        self.actions.append(action)
        next_dispatch(action)


def test_middleware_given_set_name_emits_set_numbers():
    log = Log()
    store = redux.Store(colors.reducer, middlewares=[
        colors.palettes,
        log])
    store.dispatch(colors.set_palette_name("Blues"))
    assert log.actions == [
            colors.set_palette_numbers([3, 4, 5, 6, 7, 8, 9]),
            colors.set_palette_name("Blues")]


def test_middleware_given_inconsistent_number():
    log = Log()
    store = redux.Store(colors.reducer, middlewares=[
        colors.palettes,
        log])
    actions = [
        colors.set_palette_number(256),
        colors.set_palette_name("Blues")
    ]
    for action in actions:
        store.dispatch(action)
    assert len(log.actions) == 4
    assert log.actions == [
            colors.set_palette_number(256),
            colors.set_palette_numbers([3, 4, 5, 6, 7, 8, 9]),
            colors.set_palette_number(9),
            colors.set_palette_name("Blues")]


@pytest.mark.parametrize("name,expect", [
    ("Accent", [3, 4, 5, 6, 7, 8]),
    ("Blues", [3, 4, 5, 6, 7, 8, 9]),
])
def test_palette_numbers(name, expect):
    result = colors.palette_numbers(name)
    assert result == expect


def test_controls_on_number():
    listener = unittest.mock.Mock()
    color_mapper = bokeh.models.LinearColorMapper()
    controls = colors.Controls(color_mapper)
    controls.subscribe(listener)
    controls.on_number(None, None, 5)
    listener.assert_called_once_with(colors.set_palette_number(5))


def test_controls_on_reverse():
    listener = unittest.mock.Mock()
    color_mapper = bokeh.models.LinearColorMapper()
    controls = colors.Controls(color_mapper)
    controls.subscribe(listener)
    controls.on_reverse(None, None, [0])
    listener.assert_called_once_with(colors.set_reverse(True))


@pytest.mark.parametrize("state,name,number", [
    ({}, None, None),
    ({"colorbar": {"name": "Blues", "number": 5}}, "Blues", "5")
])
def test_controls_render(state, name, number):
    color_mapper = bokeh.models.LinearColorMapper()
    controls = colors.Controls(color_mapper)
    controls.render(state)
    assert controls.dropdowns["names"].value == name
    assert controls.dropdowns["numbers"].value == number


def test_controls_render_sets_menu():
    color_mapper = bokeh.models.LinearColorMapper()
    controls = colors.Controls(color_mapper)
    names = ["A", "B"]
    numbers = [1, 2]
    state = {
        "colorbar": {"names": names, "numbers": numbers}
    }
    controls.render(state)
    assert controls.dropdowns["names"].menu == [
            ("A", "A"), ("B", "B")]
    assert controls.dropdowns["numbers"].menu == [
            ("1", "1"), ("2", "2")]



@pytest.mark.parametrize("state,palette", [
        ({}, None),
        ({"colorbar": {"name": "Accent", "number": 3}},
            ["#7fc97f", "#beaed4", "#fdc086"]),
        ({"colorbar": {"name": "Accent", "number": 3, "reverse": True}},
            ["#fdc086", "#beaed4", "#7fc97f"])
    ])
def test_controls_render_palette(state, palette):
    color_mapper = bokeh.models.LinearColorMapper()
    controls = colors.Controls(color_mapper)
    controls.render(state)
    assert color_mapper.palette == palette


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
