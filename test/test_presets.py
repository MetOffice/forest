import copy
import pytest
import unittest
import bokeh.models
from forest import presets, colors, redux


def test_preset_set_default_mode():
    state = presets.reducer({}, presets.set_default_mode())
    assert state["presets"]["meta"]["mode"] == presets.DEFAULT


def test_preset_set_edit_mode():
    state = presets.reducer({}, presets.set_edit_mode())
    assert state["presets"]["meta"]["mode"] == presets.EDIT


def test_state_to_props():
    result = presets.state_to_props({})
    expect = ([], presets.DEFAULT)
    assert expect == result


def test__reducer():
    label = "Custom-1"
    settings = {"key": "value"}
    state = {"colorbar": settings}
    action = presets.save_preset(label)
    state = presets.reducer(state, action)
    expect = {
        "labels": {0: label},
        "settings": {0: settings},
    }
    assert state["presets"] == expect


def test__reducer_save_new_preset():
    label = "Custom-2"
    settings = {"key": "value"}
    state = {
        "colorbar": settings,
        "presets": {
            "labels": {0: "Custom-1"},
            "settings": {0: settings}
        }
    }
    action = presets.save_preset("Custom-2")
    state = presets.reducer(state, action)
    result = state["presets"]
    expect = {
            "labels": {0: "Custom-1", 1: "Custom-2"},
            "settings": {0: settings, 1: settings}}
    assert expect == result


def test__reducer_load_preset():
    settings = {"key": "value"}
    state = {
        "presets": {
            "labels": {5: "Custom-1"},
            "settings": {5: settings}
        }
    }
    action = presets.load_preset("Custom-1")
    state = presets.reducer(state, action)
    assert state["colorbar"] == settings
    assert state["presets"]["active"] == 5


def test__reducer_rename_preset():
    state = {
        "presets": {
            "active": 7,
            "labels": {
                7: "Custom-1"}
        }
    }
    action = presets.rename_preset("Custom-2")
    state = presets.reducer(state, action)
    assert state["presets"]["labels"] == {7: "Custom-2"}


def test__reducer_remove_preset():
    state = {
        "presets": {
            "active": 7,
            "labels": {
                7: "Custom-1"}
        }
    }
    action = presets.remove_preset()
    state = presets.reducer(state, action)
    assert state["presets"]["labels"] == {}
    assert "active" not in state["presets"]


@pytest.mark.parametrize("call_method,action", [
    (lambda ui: ui.on_save(), presets.on_save("label")),
    (lambda ui: ui.on_load(), presets.load_preset("label"))
])
def test_ui_actions(call_method, action):
    listener = unittest.mock.Mock()
    ui = presets.PresetUI()
    ui.select.value = "label"
    ui.text_input.value = "label"
    ui.subscribe(listener)
    call_method(ui)
    listener.assert_called_once_with(action)


def test_reducer_given_empty_state():
    state = {}
    action = presets.save_preset("label")
    assert presets.reducer(state, action) == {
            "presets": {
                "labels": {0: "label"},
                "settings": {0: {}}
            }
    }


def test_reducer_save_preset_creates_presets_section():
    label = "A"
    settings = {"palette": "accent"}
    state = {"colorbar": settings}
    action = presets.save_preset(label)
    result = presets.reducer(state, action)
    uid = presets.find_id(result, label)
    assert result["colorbar"] == settings
    assert result["presets"]["labels"][uid] == label
    assert result["presets"]["settings"][uid] == settings


def test_reducer_save_preset_adds_new_entry():
    uid = 42
    state = {
        "colorbar": {"palette": "blues"},
        "presets": {
            "labels": {uid: "A"},
            "settings": {uid: {"palette": "inferno"}}
        }
    }
    action = presets.save_preset("B")
    result = presets.reducer(state, action)
    assert set(result["presets"]["labels"].values()) == {"A", "B"}
    uid = presets.find_id(result, "A")
    assert result["presets"]["settings"][uid] == {"palette": "inferno"}
    uid = presets.find_id(result, "B")
    assert result["presets"]["settings"][uid] == {"palette": "blues"}


def test_reducer_load_preset():
    uid = 42
    state = {
        "presets": {
            "labels": {uid: "A"},
            "settings": {uid: {"palette": "spectral"}}
        }
    }
    action = presets.load_preset("A")
    result = presets.reducer(state, action)
    assert result["colorbar"] == {"palette": "spectral"}


def test_reducer_save_duplicate_label():
    settings = {"palette": "Accent"}
    state = {"colorbar": settings}
    for action in [
            presets.save_preset("A"),
            presets.save_preset("A")]:
        state = presets.reducer(state, action)
    uid = presets.find_id(state, "A")
    assert set(state["presets"]["labels"].values()) == {"A"}
    assert state["presets"]["settings"][uid] == settings


def test_reducer_update():
    reducer = redux.combine_reducers(
            presets.reducer,
            colors.reducer)
    state = {}
    for action in [
            colors.set_palette_name("Viridis"),
            presets.save_preset("A"),
            colors.set_palette_name("Accent"),
            presets.save_preset("A")]:
        state = reducer(state, action)
    uid = presets.find_id(state, "A")
    assert state["colorbar"] == {"name": "Accent"}
    assert set(state["presets"]["labels"].values()) == {"A"}
    assert state["presets"]["settings"][uid] == {"name": "Accent"}


@pytest.mark.parametrize("labels,options", [
    ([], []),
    (["B", "A"], ["A", "B"])
])
def test_render(labels, options):
    ui = presets.PresetUI()
    ui.render(labels, presets.DEFAULT)
    assert ui.select.options == options
    assert isinstance(ui.select, bokeh.models.Select)
    assert isinstance(ui.buttons["save"], bokeh.models.Button)
    assert isinstance(ui.layout, bokeh.layouts.Box)


def test_render_edit_mode():
    ui = presets.PresetUI()
    ui.render([], presets.EDIT)
    row = ui.rows["content"]
    assert isinstance(row.children[0], bokeh.models.TextInput)
    assert isinstance(row.children[1], bokeh.models.Button)
    assert isinstance(row.children[2], bokeh.models.Button)
