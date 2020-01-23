import copy
import pytest
import unittest
import bokeh.models
from forest import presets, colors, redux


@pytest.fixture
def store():
    return redux.Store(presets.reducer)


@pytest.mark.parametrize("action,expect", [
    (presets.on_edit(), [
        presets.set_edit_label(""),
        presets.set_edit_mode()
    ]),
    (presets.on_new(), [
        presets.set_edit_label(""),
        presets.set_edit_mode()
    ]),
])
def test_middleware(store, action, expect):
    result = list(presets.middleware(store, action))
    assert expect == result


def test_reducer_set_labels():
    labels = ["A"]
    state = presets.reducer({}, presets.set_labels(labels))
    assert set(labels) == set(state["presets"]["labels"].values())


@pytest.mark.parametrize("state,expect", [
    ({}, ""),
    ({"presets": {"labels": {3: "L"}}}, ""),
    ({"presets": {"active": 3, "labels": {3: "L"}}}, "L")
])
def test_query_label(state, expect):
    result = presets.Query(state).label
    assert expect == result


def test_preset_set_default_mode():
    state = presets.reducer({}, presets.set_default_mode())
    assert state["presets"]["meta"]["mode"] == presets.DEFAULT


def test_preset_set_edit_mode():
    state = presets.reducer({}, presets.set_edit_mode())
    assert state["presets"]["meta"]["mode"] == presets.EDIT


def test_preset_set_edit_label():
    state = presets.reducer({}, presets.set_edit_label("Label"))
    assert state["presets"]["meta"]["label"] == "Label"


def test_state_to_props():
    result = presets.state_to_props({})
    expect = ([], presets.DEFAULT, "")
    assert expect == result


def test_reducer_save_preset():
    label = "Custom-1"
    state = {"colorbar": {"key": "value"}}
    action = presets.save_preset(label)
    state = presets.reducer(state, action)
    expect = {"labels": {0: label}}
    assert state["presets"] == expect


def test_reducer_save_new_preset():
    label = "Custom-2"
    state = {
        "colorbar": {"key": "value"},
        "presets": {
            "labels": {0: "Custom-1"}
        }
    }
    action = presets.save_preset("Custom-2")
    state = presets.reducer(state, action)
    result = state["presets"]
    expect = {"labels": {0: "Custom-1", 1: "Custom-2"}}
    assert expect == result


def test__reducer_load_preset():
    settings = {"key": "value"}
    state = {
        "presets": {
            "labels": {5: "Custom-1"},
        }
    }
    action = presets.load_preset("Custom-1")
    state = presets.reducer(state, action)
    assert state["presets"]["active"] == 5


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
    (lambda ui: ui.on_load(None, None, "label"), presets.on_load("label"))
])
def test_ui_actions(call_method, action):
    listener = unittest.mock.Mock()
    ui = presets.PresetUI()
    ui.select.value = "label"
    ui.text_input.value = "label"
    ui.add_subscriber(listener)
    call_method(ui)
    listener.assert_called_once_with(action)


def test_reducer_given_empty_state():
    state = {}
    action = presets.save_preset("label")
    assert presets.reducer(state, action) == {
            "presets": {
                "labels": {0: "label"}
            }
    }


def test_reducer_save_preset_creates_presets_section():
    label = "A"
    state = {"colorbar": {}}
    action = presets.save_preset(label)
    result = presets.reducer(state, action)
    uid = presets.Query(result).find_id(label)
    assert result["presets"]["labels"][uid] == label


def test_reducer_save_preset_adds_new_entry():
    uid = 42
    state = {
        "presets": {
            "labels": {uid: "A"},
        }
    }
    action = presets.save_preset("B")
    result = presets.reducer(state, action)
    assert set(result["presets"]["labels"].values()) == {"A", "B"}


def test_reducer_update():
    reducer = redux.combine_reducers(
            presets.reducer,
            colors.reducer)
    state = {}
    for action in [
            presets.save_preset("A"),
            presets.save_preset("A")]:
        state = reducer(state, action)
    assert set(state["presets"]["labels"].values()) == {"A"}


@pytest.mark.parametrize("labels,options", [
    ([], []),
    (["B", "A"], ["A", "B"])
])
def test_render(labels, options):
    ui = presets.PresetUI()
    ui.render(labels, presets.DEFAULT, "")
    assert ui.select.options == options
    assert isinstance(ui.select, bokeh.models.Select)
    assert isinstance(ui.buttons["save"], bokeh.models.Button)
    assert isinstance(ui.layout, bokeh.layouts.Box)


def test_render_edit_mode():
    ui = presets.PresetUI()
    ui.render([], presets.EDIT, "Name")
    row = ui.rows["content"]
    assert isinstance(row.children[0], bokeh.models.TextInput)
    assert isinstance(row.children[1], bokeh.models.Button)
    assert isinstance(row.children[2], bokeh.models.Button)
    assert ui.text_input.value == "Name"
