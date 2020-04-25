import unittest.mock
from unittest.mock import Mock, sentinel
import pytest
import forest
import forest.components
from forest import colors, main, redux, db
import bokeh.models
import numpy as np


@pytest.fixture
def listener():
    return unittest.mock.Mock()


@pytest.mark.parametrize("state,action,expect", [
    ({}, colors.set_palette_name("Accent"), {"colorbar": {"name": "Accent"}}),
    ({}, colors.set_palette_number(3), {"colorbar": {"number": 3}}),
    ({}, colors.set_palette_numbers([1, 2 ,3]), {"colorbar": {"numbers": [1, 2, 3]}}),
    pytest.param({}, colors.set_user_high(100), {
        "colorbar": {
            "limits": {
                "user": {
                    "high": 100
                }
            }
        }
    }, id="set user high"),
    pytest.param({}, colors.set_user_low(0), {
        "colorbar": {
            "limits": {
                "user": {
                    "low": 0
                }
            }
        }
    }, id="set user low"),
    pytest.param({}, colors.set_source_limits(0, 100), {
        "colorbar": {
            "limits": {
                "column_data_source": {
                    "low": 0,
                    "high": 100
                }
            }
        }
    }, id="set source limits"),
    ({}, colors.set_colorbar({"key": "value"}), {"colorbar": {"key": "value"}}),
    ({}, colors.set_palette_names(["example"]), {"colorbar": {"names": ["example"]}}),
    ({}, colors.set_edit_layer("Foo"), {"colorbar": {"edit": "Foo"}}),
])
def test_reducer_sets_colorbar_namespace(state, action, expect):
    result = forest.reducer(state, action)
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


@pytest.mark.parametrize("actions,expect", [
    pytest.param([
        colors.set_edit_layer("Foo"),
        colors.set_palette_name("Accent"),
        colors.set_edit_layer("Bar"),
        colors.set_palette_name("Blues"),
    ], {
        "edit": "Bar",
        "layers": {
            "Foo": {
                "colorbar": {"name": "Accent"}
            },
            "Bar": {
                "colorbar": {"name": "Blues"}
            }
        }
    })
])
def test_reducer_multiple_actions(actions, expect):
    state = {}
    for action in actions:
        state = colors.reducer(state, action)
    assert expect == state["colorbar"]


def test_reducer_immutable_state():
    state = {"colorbar": {"number": 1}}
    colors.reducer(state, colors.set_palette_number(3))
    assert state["colorbar"]["number"] == 1


@pytest.mark.parametrize("state,expect", [
    pytest.param(
        {}, forest.colors.ColorSpec()),
    pytest.param(
        {"colorbar": {"name": "Accent"}},
        forest.colors.ColorSpec(name="Accent")),
    pytest.param(
        {"colorbar": {"name": "Accent", "numbers": [1, 2, 3]}},
        forest.colors.ColorSpec(name="Accent")),
    pytest.param(
        {
            "colorbar": {
                "layers": {
                    "label": {"colorbar": {"name": "Blues"}}
                },
                "name": "Accent", "numbers": [1, 2, 3]}
        },
        forest.colors.ColorSpec(name="Blues")),
])
def test_spec_parser(state, expect):
    spec_parser = forest.colors.SpecParser("label")
    assert spec_parser(state) == expect


@pytest.fixture
def color_view():
    color_mapper = bokeh.models.LinearColorMapper()
    spec_parser = forest.colors.SpecParser("label")
    return forest.colors.ColorView(color_mapper, spec_parser)


def test_color_view_render(color_view):
    color_view.render({"colorbar": {"low": 42}})
    assert color_view.color_mapper.low == 42


def test_color_view_connect(color_view):
    store = unittest.mock.Mock()
    color_view.connect(store)
    store.add_subscriber.assert_called_once_with(color_view.render)


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


def test_colorbar_spec():
    colorbar_spec = colors.ColorSpec()
    assert colorbar_spec.nan_color.to_css() == "rgba(0, 0, 0, 0)"


def test_colorbar_spec_apply():
    color_mapper = bokeh.models.LinearColorMapper()
    colorbar_spec = colors.ColorSpec()
    colorbar_spec.apply(color_mapper)
    assert color_mapper.nan_color.to_css() == "rgba(0, 0, 0, 0)"


def test_colorbars_palette():
    colorbars = forest.components.Colorbars()
    colorbars.render({"name": "Accent", "number": 3})
    assert colorbars.color_mapper.palette == ['#7fc97f', '#beaed4', '#fdc086']


def test_controls_on_name(listener):
    controls = colors.ColorbarControls()
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


@pytest.mark.parametrize("name,expect", [
    ("Accent", [3, 4, 5, 6, 7, 8]),
    ("Viridis", [3, 4, 5, 6, 7, 8, 9, 10, 11, 256]),
])
def test_palette_numbers(name, expect):
    result = colors.palette_numbers(name)
    assert result == expect


def test_controls_on_number(listener):
    controls = colors.ColorbarControls()
    controls.add_subscriber(listener)
    controls.on_number(None, None, 5)
    listener.assert_called_once_with(colors.set_palette_number(5))


def test_controls_on_reverse(listener):
    controls = colors.ColorbarControls()
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
    controls = colors.ColorbarControls()
    controls.render(props)
    assert controls.dropdowns[key].label == label


def test_controls_render_sets_menu():
    controls = colors.ColorbarControls()
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
            ["#7fc97f", "#beaed4", "#fdc086"]),
        ({"name": "Accent", "number": 3, "reverse": True},
            ["#fdc086", "#beaed4", "#7fc97f"])
    ])
def test_controls_render_palette(props, palette):
    controls = forest.components.Colorbars()
    controls.render(props)
    assert controls.color_mapper.palette == palette


@pytest.mark.parametrize("props,active", [
    ({}, []),
    ({"reverse": False}, []),
    ({"reverse": True}, [0]),
])
def test_color_palette_render_checkbox(props, active):
    color_palette = colors.ColorbarControls()
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
    controls = colors.ColorbarControls()
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
    controls = colors.ColorbarControls()
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
