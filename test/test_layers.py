import pytest
import unittest.mock
import bokeh.plotting
from forest import layers, redux


@pytest.mark.parametrize("action,expect", [
    ({"kind": "ANY"}, [{"kind": "ANY"}]),
    (layers.on_dropdown(0, "label"), [layers.set_label(0, "label")]),
    (layers.on_button_group(0, [0, 1, 2]), [layers.set_active(0, [0, 1, 2])]),
])
def test_middleware(action, expect):
    store = redux.Store(layers.reducer)
    result = list(layers.middleware(store, action))
    assert expect == result


@pytest.fixture
def listener():
    return unittest.mock.Mock()


def test_figure_dropdown(listener):
    ui = layers.FigureUI()
    ui.add_subscriber(listener)
    ui.on_change(None, None, ui.labels[0])
    listener.assert_called_once_with(layers.set_figures(1))


def test_remove(listener):
    ui = layers.LayersUI([])
    ui.add_subscriber(listener)
    ui.on_click_remove()
    listener.assert_called_once_with(layers.on_remove())


@pytest.mark.parametrize("state,actions,expect", [
    ({}, layers.set_figures(3), {"figures": 3}),
    ({}, layers.add_layer("Name"), {"labels": ["Name"]}),
    ({}, layers.on_remove(), {"labels": []}),
    ({"layers": {"labels": [None, None]}}, layers.on_remove(), {"labels": [None]}),
    ({}, layers.set_label(0, "Label"), {"labels": ["Label"]}),
    ({}, layers.set_label(0, "Other"), {"labels": ["Other"]}),
    ({}, layers.set_label(1, "Label"), {"labels": [None, "Label"]}),
    ({}, layers.set_label(2, "Label"), {"labels": [None, None, "Label"]}),
    ({}, layers.set_active(0, []), {"active": [[]]}),
    ({}, layers.set_active(1, []), {"active": [None, []]}),
    ({"layers": {"active": [[]]}}, layers.set_active(0, [0]), {"active": [[0]]}),
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


def test_controls_render_sets_button_groups():
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
    assert controls.button_groups[0].active == [0, 1, 2]
    assert controls.button_groups[0].labels == ["L", "C", "R"]


@pytest.mark.parametrize("from_labels,to_labels,expect", [
    ([], [], 0),
    ([], ["label"], 1),
    ([], ["label", "label"], 2),
    (["label"], [], 0),
    (["label", "label"], [], 0),
    (["label", "label"], ["label"], 1),
])
def test_controls_render_number_of_rows(from_labels, to_labels, expect):
    from_state = {
        "layers": {"labels": from_labels}
    }
    to_state = {
        "layers": {"labels": to_labels}
    }
    controls = layers.LayersUI([])
    controls.render(*controls.to_props(from_state))
    controls.render(*controls.to_props(to_state))
    assert len(controls.columns["rows"].children) == expect


def test_on_button_group(listener):
    row_index = 3
    attr, old, new = None, [], [2]
    controls = layers.LayersUI([])
    controls.add_subscriber(listener)
    controls.on_button_group(row_index)(attr, old, new)
    listener.assert_called_once_with(layers.on_button_group(row_index, new))


def test_on_dropdown(listener):
    row_index = 3
    attr, old, new = None, None, "label"
    controls = layers.LayersUI([])
    controls.add_subscriber(listener)
    controls.on_dropdown(row_index)(attr, old, new)
    listener.assert_called_once_with(layers.on_dropdown(row_index, new))


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


@pytest.mark.parametrize("n", [1, 2, 3])
def test_figure_row(n):
    figures = [bokeh.plotting.figure() for _ in range(3)]
    figure_row = layers.FigureRow(figures)
    figure_row.render(n)
    assert len(figure_row.layout.children) == n
