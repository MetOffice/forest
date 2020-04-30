import unittest.mock
from unittest.mock import Mock, sentinel
import pytest
from forest import colors, main, redux, db
import bokeh.models
import numpy as np


@pytest.mark.parametrize("given,expect", [
    pytest.param({}, colors.ColorSpec()),
    pytest.param({
        "limits": {
            "origin": "user",
            "user": {
                "high": 100,
                "low": 42
            }
        }
    }, colors.ColorSpec(low=42, high=100)),
    pytest.param({
        "limits": {
            "origin": "column_data_source",
            "user": {
                "high": 100,
                "low": 42
            },
            "column_data_source": {
                "high": 5,
                "low": 4
            }
        }
    }, colors.ColorSpec(low=4, high=5)),
    pytest.param({
        "name": "Accent",
        "number": 3,
        "reverse": True,
        "invisible_min": True,
        "invisible_max": True,
    }, colors.ColorSpec(name="Accent",
                        number=3,
                        reverse=True,
                        low_visible=False,
                        high_visible=False)),
])
def test_parse_color_spec(given, expect):
    assert colors.parse_color_spec(given) == expect


@pytest.fixture
def listener():
    return unittest.mock.Mock()


@pytest.mark.parametrize("state,action,expect", [
    ({}, colors.set_palette_name("Accent"), {"colorbar": {"name": "Accent"}}),
    ({}, colors.set_palette_number(3), {"colorbar": {"number": 3}}),
    ({}, colors.set_palette_numbers([1, 2 ,3]), {"colorbar": {"numbers": [1, 2, 3]}}),
    ({}, colors.set_user_high(100), {}),
    ({}, colors.set_user_low(0), {}),
    ({}, colors.set_source_limits(0, 100), {}),
    ({}, colors.set_colorbar({"key": "value"}), {"colorbar": {"key": "value"}}),
    ({}, colors.set_palette_names(["example"]), {"colorbar": {"names": ["example"]}}),
])
def test_reducer(state, action, expect):
    result = colors.reducer(state, action)
    assert expect == result


def test_limits_reducer_origin():
    text = "foo"
    action = colors.set_limits_origin(text)
    state = colors.limits_reducer({}, action)
    assert state["colorbar"]["limits"]["origin"] == text


def test_limits_reducer_user_low():
    number = 42
    action = colors.set_user_low(number)
    state = colors.limits_reducer({}, action)
    origin = "user"
    assert state["colorbar"]["limits"][origin]["low"] == number


def test_limits_reducer_user_high():
    number = 42
    action = colors.set_user_high(number)
    state = colors.limits_reducer({}, action)
    origin = "user"
    assert state["colorbar"]["limits"][origin]["high"] == number


def test_limits_reducer_source():
    low, high = 42, 1729
    action = colors.set_source_limits(low, high)
    state = colors.limits_reducer({}, action)
    origin = "column_data_source"
    assert state["colorbar"]["limits"][origin]["low"] == low
    assert state["colorbar"]["limits"][origin]["high"] == high


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
        "reverse": False,
        "invisible_min": False,
        "invisible_max": False
    }
    assert colors.defaults() == expected


def test_color_controls():
    color_mapper = bokeh.models.LinearColorMapper()
    controls = colors.ColorMapperView(color_mapper)
    controls.render({"name": "Accent", "number": 3})
    assert color_mapper.palette == ('#7fc97f', '#beaed4', '#fdc086')


def test_controls_on_name(listener):
    event = Mock()
    event.item = 5
    controls = colors.ColorPalette()
    controls.add_subscriber(listener)
    controls.on_number(event)
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


@pytest.mark.parametrize("name,expect", [
    ("Accent", [3, 4, 5, 6, 7, 8]),
    ("Viridis", [3, 4, 5, 6, 7, 8, 9, 10, 11, 256]),
])
def test_palette_numbers(name, expect):
    result = colors.palette_numbers(name)
    assert result == expect


def test_controls_on_number(listener):
    event = Mock()
    event.item = 5
    controls = colors.ColorPalette()
    controls.add_subscriber(listener)
    controls.on_number(event)
    listener.assert_called_once_with(colors.set_palette_number(5))


def test_controls_on_reverse(listener):
    controls = colors.ColorPalette()
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
    controls = colors.ColorPalette()
    controls.render(props)
    assert controls.dropdowns[key].label == label


def test_controls_render_sets_menu():
    controls = colors.ColorPalette()
    names = ["A", "B"]
    numbers = [1, 2]
    props = {"names": names, "numbers": numbers}
    controls.render(props)
    assert controls.dropdowns["names"].menu == [
            ("A", "A"), ("B", "B")]
    assert controls.dropdowns["numbers"].menu == [
            ("1", "1"), ("2", "2")]


@pytest.mark.parametrize("props,palette", [
        ({"name": "Accent", "number": 3},
            ("#7fc97f", "#beaed4", "#fdc086")),
        ({"name": "Accent", "number": 3, "reverse": True},
            ("#fdc086", "#beaed4", "#7fc97f"))
    ])
def test_controls_render_palette(props, palette):
    color_mapper = bokeh.models.LinearColorMapper()
    controls = colors.ColorMapperView(color_mapper)
    controls.render(props)
    assert color_mapper.palette == palette


@pytest.mark.parametrize("props,active", [
    ({}, []),
    ({"reverse": False}, []),
    ({"reverse": True}, [0]),
])
def test_color_palette_render_checkbox(props, active):
    color_palette = colors.ColorPalette()
    color_palette.render(props)
    assert color_palette.checkbox.active == active


@pytest.mark.parametrize("key,props,active", [
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
    user_limits.render({"limits": {"user": {"low": -1, "high": 1}}})
    assert user_limits.inputs["low"].value == "-1"
    assert user_limits.inputs["high"].value == "1"


@pytest.mark.parametrize("sources,low,high", [
    ([], 0, 1),
    ([
        bokeh.models.ColumnDataSource(
            {"image": [np.linspace(-1, 1, 4).reshape(2, 2)]}
        )], -1, 1)
])
def test_source_limits_on_change(listener, sources, low, high):
    source_limits = colors.SourceLimits()
    for source in sources:
        source_limits.add_source(source)
    source_limits.add_subscriber(listener)
    source_limits.on_change(None, None, None)  # attr, old, new
    listener.assert_called_once_with(colors.set_source_limits(low, high))


def test_remove_source():
    """Should be able to stop listening to changes"""
    source = bokeh.models.ColumnDataSource()
    limits = colors.SourceLimits()
    limits.add_source(source)
    limits.remove_source(source)
    assert source not in limits.sources


def test_render_called_once_with_two_identical_settings():
    store = redux.Store(colors.reducer)
    controls = colors.ColorPalette()
    controls.render = unittest.mock.Mock()
    controls.connect(store)
    for action in [
            colors.set_palette_name("Accent"),
            colors.set_palette_name("Accent")]:
        store.dispatch(action)
    controls.render.assert_called_once()


def test_render_called_once_with_non_relevant_settings():
    """Render should only happen when relevant state changes"""
    store = redux.Store(
            redux.combine_reducers(
                db.reducer,
                colors.reducer))
    controls = colors.ColorPalette()
    controls.render = unittest.mock.Mock()
    controls.connect(store)
    for action in [
            colors.set_palette_name("Accent"),
            db.set_value("variable", "air_temperature")]:
        store.dispatch(action)
    controls.render.assert_called_once()


def test_limits_middleware():
    # Swallow actions that don't change state
    middleware = colors.middleware()
    store = None
    result = []
    actions =[
        colors.set_user_low(0.0000001),
        colors.set_user_low(0.0000001)]
    for action in actions:
        result += list(middleware(store, action))
    assert result == [actions[0]]
