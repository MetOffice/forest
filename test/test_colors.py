import unittest.mock
import pytest
from forest import colors, main, redux, db
import bokeh.models
import numpy as np


@pytest.fixture
def listener():
    return unittest.mock.Mock()


@pytest.mark.parametrize("state,action,expect", [
    ({}, colors.set_fixed(True), {"colorbar": {"fixed": True}}),
    ({}, colors.set_fixed(False), {"colorbar": {"fixed": False}}),
    ({}, colors.set_palette_name("Accent"), {"colorbar": {"name": "Accent"}}),
    ({}, colors.set_palette_number(3), {"colorbar": {"number": 3}}),
    ({}, colors.set_palette_numbers([1, 2 ,3]), {"colorbar": {"numbers": [1, 2, 3]}}),
    ({}, colors.set_user_high(100), {"colorbar": {"high": 100}}),
    ({}, colors.set_user_low(0), {"colorbar": {"low": 0}}),
    ({}, colors.set_source_limits(0, 100), {"colorbar": {"low": 0, "high": 100}}),
    ({}, colors.set_colorbar({"key": "value"}), {"colorbar": {"key": "value"}}),
    ({}, colors.set_palette_names(["example"]), {"colorbar": {"names": ["example"]}}),
])
def test_reducer(state, action, expect):
    result = colors.reducer(state, action)
    assert expect == result


def test_reducer_immutable_state():
    state = {"colorbar": {"number": 1}}
    colors.reducer(state, colors.set_palette_number(3))
    assert state["colorbar"]["number"] == 1


def test_defaults():
    expected = {
        "name": "Viridis",
        "number": 256,
        "names": colors.palette_names(),
        "numbers": colors.palette_numbers("Viridis"),
        "low": 0,
        "high": 1,
        "fixed": False,
        "reverse": False,
        "invisible_min": False,
        "invisible_max": False
    }
    assert colors.defaults() == expected


def test_color_controls():
    color_mapper = bokeh.models.LinearColorMapper()
    controls = colors.ColorPalette(color_mapper)
    controls.render({"name": "Accent", "number": 3})
    assert color_mapper.palette == ['#7fc97f', '#beaed4', '#fdc086']


def test_controls_on_name(listener):
    color_mapper = bokeh.models.LinearColorMapper()
    controls = colors.ColorPalette(color_mapper)
    controls.add_subscriber(listener)
    controls.on_number(None, None, 5)
    listener.assert_called_once_with(colors.set_palette_number(5))


def test_middleware_given_empty_state_emits_set_colorbar():
    store = redux.Store(colors.reducer)
    action = {"kind": "ANY"}
    result = list(colors.palettes(store, action))
    expect = [action, colors.set_colorbar(colors.defaults())]
    assert expect == result


def test_middleware_given_incomplete_state_emits_set_colorbar():
    store = redux.Store(colors.reducer, initial_state={"colorbar": {"low": -1}})
    action = {"kind": "ANY"}
    result = list(colors.palettes(store, action))
    settings = {**colors.defaults(), **{"low": -1}}
    expect = [action, colors.set_colorbar(settings)]
    assert expect == result


def test_middleware_given_set_name_emits_set_numbers():
    store = redux.Store(colors.reducer)
    action = colors.set_palette_name("Blues")
    result = list(colors.palettes(store, action))
    assert result == [
            colors.set_palette_numbers(
                colors.palette_numbers("Blues")),
            colors.set_palette_name("Blues")]


def test_middleware_given_inconsistent_number():
    store = redux.Store(colors.reducer)
    store.dispatch(colors.set_palette_number(1000))
    action = colors.set_palette_name("Viridis")
    result = list(colors.palettes(store, action))
    assert result == [
            colors.set_palette_numbers(
                colors.palette_numbers("Viridis")),
            colors.set_palette_number(256),
            colors.set_palette_name("Viridis")]


def test_middleware_given_fixed_swallows_source_limit_actions():
    store = redux.Store(colors.reducer, middlewares=[
        colors.palettes])
    store.dispatch(colors.set_fixed(True))
    action = colors.set_source_limits(0, 100)
    assert list(colors.palettes(store, action)) == []
    assert store.state == {"colorbar": {"fixed": True}}


def test_middleware_given_fixed_allows_source_limit_actions():
    store = redux.Store(colors.reducer)
    action = colors.set_source_limits(0, 100)
    assert list(colors.palettes(store, action)) == [action]


@pytest.mark.parametrize("name,expect", [
    ("Accent", [3, 4, 5, 6, 7, 8]),
    ("Viridis", [3, 4, 5, 6, 7, 8, 9, 10, 11, 256]),
])
def test_palette_numbers(name, expect):
    result = colors.palette_numbers(name)
    assert result == expect


def test_controls_on_number(listener):
    color_mapper = bokeh.models.LinearColorMapper()
    controls = colors.ColorPalette(color_mapper)
    controls.add_subscriber(listener)
    controls.on_number(None, None, 5)
    listener.assert_called_once_with(colors.set_palette_number(5))


def test_controls_on_reverse(listener):
    color_mapper = bokeh.models.LinearColorMapper()
    controls = colors.ColorPalette(color_mapper)
    controls.add_subscriber(listener)
    controls.on_reverse(None, None, [0])
    listener.assert_called_once_with(colors.set_reverse(True))


@pytest.mark.parametrize("key,props,label", [
    ("numbers", {}, "N"),
    ("names", {}, "Palettes"),
    ("numbers", {"name": "Blues", "number": 5}, "5"),
    ("names", {"name": "Blues", "number": 5}, "Blues")
])
def test_controls_render_label(key, props, label):
    color_mapper = bokeh.models.LinearColorMapper()
    controls = colors.ColorPalette(color_mapper)
    controls.render(props)
    assert controls.dropdowns[key].label == label


def test_controls_render_sets_menu():
    color_mapper = bokeh.models.LinearColorMapper()
    controls = colors.ColorPalette(color_mapper)
    names = ["A", "B"]
    numbers = [1, 2]
    props = {"names": names, "numbers": numbers}
    controls.render(props)
    assert controls.dropdowns["names"].menu == [
            ("A", "A"), ("B", "B")]
    assert controls.dropdowns["numbers"].menu == [
            ("1", "1"), ("2", "2")]


@pytest.mark.parametrize("props,palette", [
        ({}, None),
        ({"name": "Accent", "number": 3},
            ["#7fc97f", "#beaed4", "#fdc086"]),
        ({"name": "Accent", "number": 3, "reverse": True},
            ["#fdc086", "#beaed4", "#7fc97f"])
    ])
def test_controls_render_palette(props, palette):
    color_mapper = bokeh.models.LinearColorMapper()
    controls = colors.ColorPalette(color_mapper)
    controls.render(props)
    assert color_mapper.palette == palette


@pytest.mark.parametrize("props,active", [
    ({}, []),
    ({"reverse": False}, []),
    ({"reverse": True}, [0]),
])
def test_color_palette_render_checkbox(props, active):
    color_mapper = bokeh.models.LinearColorMapper()
    color_palette = colors.ColorPalette(color_mapper)
    color_palette.render(props)
    assert color_palette.checkbox.active == active


@pytest.mark.parametrize("key,props,active", [
    ("fixed", {}, []),
    ("fixed", {"fixed": False}, []),
    ("fixed", {"fixed": True}, [0]),
    ("invisible_min", {}, []),
    ("invisible_min", {"invisible_min": False}, []),
    ("invisible_min", {"invisible_min": True}, [0]),
    ("invisible_max", {}, []),
    ("invisible_max", {"invisible_max": False}, []),
    ("invisible_max", {"invisible_max": True}, [0]),
])
def test_user_limits_render_checkboxes(key, props, active):
    user_limits = colors.UserLimits()
    user_limits.render(props)
    assert user_limits.checkboxes[key].active == active


def test_user_limits_render():
    user_limits = colors.UserLimits()
    user_limits.render({"low": -1, "high": 1})
    assert user_limits.inputs["low"].value == "-1"
    assert user_limits.inputs["high"].value == "1"


@pytest.mark.parametrize("new,action", [
    ([0], colors.set_fixed(True)),
    ([], colors.set_fixed(False)),
])
def test_user_limits_on_fixed(listener, new, action):
    user_limits = colors.UserLimits()
    user_limits.add_subscriber(listener)
    user_limits.on_checkbox_change(None, None, new)
    listener.assert_called_once_with(action)


@pytest.mark.parametrize("sources,low,high", [
    ([], 0, 1),
    ([
        bokeh.models.ColumnDataSource(
            {"image": [np.linspace(-1, 1, 4).reshape(2, 2)]}
        )], -1, 1)
])
def test_source_limits_on_change(listener, sources, low, high):
    source_limits = colors.SourceLimits(sources)
    source_limits.add_subscriber(listener)
    source_limits.on_change(None, None, None)  # attr, old, new
    listener.assert_called_once_with(colors.set_source_limits(low, high))


def test_render_called_once_with_two_identical_settings():
    color_mapper = bokeh.models.LinearColorMapper()
    store = redux.Store(colors.reducer)
    controls = colors.ColorPalette(color_mapper)
    controls.render = unittest.mock.Mock()
    controls.connect(store)
    for action in [
            colors.set_palette_name("Accent"),
            colors.set_palette_name("Accent")]:
        store.dispatch(action)
    controls.render.assert_called_once()


def test_render_called_once_with_non_relevant_settings():
    """Render should only happen when relevant state changes"""
    color_mapper = bokeh.models.LinearColorMapper()
    store = redux.Store(
            redux.combine_reducers(
                db.reducer,
                colors.reducer))
    controls = colors.ColorPalette(color_mapper)
    controls.render = unittest.mock.Mock()
    controls.connect(store)
    for action in [
            colors.set_palette_name("Accent"),
            db.set_value("variable", "air_temperature")]:
        store.dispatch(action)
    controls.render.assert_called_once()

