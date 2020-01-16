import pytest
import unittest.mock
from forest import layers, redux


def test_middleware():
    store = redux.Store(layers.reducer)
    action = {"kind": "ANY"}
    result = list(layers.middleware(store, action))
    expect = [action]
    assert expect == result


@pytest.fixture
def listener():
    return unittest.mock.Mock()


def test_figure_dropdown(listener):
    ui = layers.FigureUI()
    ui.subscribe(listener)
    ui.on_change(None, None, ui.labels[0])
    listener.assert_called_once_with(layers.set_figures(1))


def test_add(listener):
    ui = layers.LayersUI([])
    ui.subscribe(listener)
    ui.on_click_add()
    listener.assert_called_once_with(layers.on_add())


def test_remove(listener):
    ui = layers.LayersUI([])
    ui.subscribe(listener)
    ui.on_click_remove()
    listener.assert_called_once_with(layers.on_remove())


@pytest.mark.parametrize("state,actions,expect", [
    ({}, layers.set_figures(3), {"figures": 3}),
    ({}, layers.on_add(), {"labels": [None]}),
    ({}, [layers.on_add(), layers.on_add()], {"labels": [None, None]}),
    ({}, layers.on_remove(), {"labels": []}),
    ({"layers": {"labels": [None, None]}}, layers.on_remove(), {"labels": [None]}),
    ({}, layers.set_label(0, "Label"), {"labels": ["Label"]}),
    ({}, layers.set_label(0, "Other"), {"labels": ["Other"]}),
    ({}, layers.set_label(1, "Label"), {"labels": [None, "Label"]}),
    ({}, layers.set_label(2, "Label"), {"labels": [None, None, "Label"]}),
    ({}, layers.set_active(0, []), {"active": [[]]}),
])
def test_reducer(state, actions, expect):
    if isinstance(actions, dict):
        actions = [actions]
    for action in actions:
        state = layers.reducer(state, action)
    result = state["layers"]
    assert result == expect


def test_controls_render():
    state = {
        "layers": {
            "labels": [None, "B", "C"]
        }
    }
    controls = layers.LayersUI([])
    controls.render(*controls.to_props(state))
    assert controls.dropdowns[0].label == "Model/observation"
    assert controls.dropdowns[1].label == "B"
    assert controls.dropdowns[2].label == "C"


def test_controls_render_sets_radio_buttons():
    state = {
        "layers": {
            "figures": 3,
            "labels": [None],
            "active": [[0, 1, 2]]
        }
    }
    controls = layers.LayersUI([])
    controls.render(*controls.to_props(state))
    assert controls.dropdowns[0].label == "Model/observation"
    assert controls.groups[0].active == [0, 1, 2]
    assert controls.groups[0].labels == ["L", "C", "R"]


def test_on_radio_button(listener):
    row_index = 3
    attr, old, new = None, [], [2]
    controls = layers.LayersUI([])
    controls.subscribe(listener)
    controls.on_radio_button(row_index)(attr, old, new)
    listener.assert_called_once_with(layers.on_radio_button(row_index, new))


@pytest.mark.parametrize("labels,active_list,expect", [
    ([], [], {}),
    (["label"], [[0]], {"label": [True, False, False]}),
    (["A", "A"],[[0], [2]], {"A": [True, False, True]}),
    (["A", None],[[0], [2]], {"A": [True, False, False]}),
])
def test_ui_state_to_visible_state(labels, active_list, expect):
    result = layers.to_visible_state(labels, active_list)
    assert expect == result


@pytest.mark.parametrize("left,right,expect", [
    ({}, {}, []),
    ({}, {"label": [True, False, False]}, [
        ("label", 0, True),
        ("label", 1, False),
        ("label", 2, False),
    ]),
    ({"label": [True, False, False]}, {}, [
        ("label", 0, False),
    ]),
    ({"label": [True, False, False]}, {"label": [False, False, False]}, [
        ("label", 0, False),
    ]),
    ({"label": [False, False, False]}, {"label": [False, True, False]}, [
        ("label", 1, True),
    ]),
])
def test_diff_visible_states(left, right, expect):
    """Needed to be efficient when toggling renderer.visible and calling
       viewer.render(tuple_state)"""
    result = layers.diff_visible_states(left, right)
    assert expect == result
