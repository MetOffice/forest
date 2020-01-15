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
    ui = layers.Controls([])
    ui.subscribe(listener)
    ui.on_click_add()
    listener.assert_called_once_with(layers.on_add())


def test_remove(listener):
    ui = layers.Controls([])
    ui.subscribe(listener)
    ui.on_click_remove()
    listener.assert_called_once_with(layers.on_remove())


@pytest.mark.parametrize("state,actions,expect", [
    ({}, layers.set_figures(3), {"figures": 3}),
    ({}, layers.on_add(), {"layers": [None]}),
    ({}, [layers.on_add(), layers.on_add()], {"layers": [None, None]}),
    ({}, layers.on_remove(), {"layers": []}),
    ({"layers": [None, None]}, layers.on_remove(), {"layers": [None]}),
    ({}, layers.set_label(0, "Label"), {"layers": ["Label"]}),
    ({}, layers.set_label(0, "Other"), {"layers": ["Other"]}),
    ({}, layers.set_label(1, "Label"), {"layers": [None, "Label"]}),
    ({}, layers.set_label(2, "Label"), {"layers": [None, None, "Label"]}),
    (
        {},
        layers.set_visible({"label": [False, False, False]}),
        {"visible": {"label": [False, False, False]}},
    ), (
        {"visible": {"other": [False, False, False]}},
        layers.set_visible({"label": [False, False, False]}),
        {"visible": {
            "other": [False, False, False],
            "label": [False, False, False]}},
    )
])
def test_reducer(state, actions, expect):
    if isinstance(actions, dict):
        actions = [actions]
    for action in actions:
        state = layers.reducer(state, action)
    assert state == expect


def test_controls_render():
    state = {"layers": [None, "B", "C"]}
    controls = layers.Controls([])
    controls.render(*controls.to_props(state))
    assert controls.dropdowns[0].label == "Model/observation"
    assert controls.dropdowns[1].label == "B"
    assert controls.dropdowns[2].label == "C"


def test_on_radio_button(listener):
    row_index = 3
    attr, old, new = None, [], [2]
    controls = layers.Controls([])
    controls.subscribe(listener)
    controls.on_radio_button(row_index)(attr, old, new)
    listener.assert_called_once_with(layers.on_radio_button(row_index, new))


@pytest.mark.parametrize("ui_state,expect", [
    ({}, {}),
    ({"layers": ["label"], "visible": [[0]]},
     {"label": [True, False, False]}),
    ({"layers": ["A", "A"], "visible": [[0], [2]]},
     {"A": [True, False, True]})
])
def test_ui_state_to_visible_state(ui_state, expect):
    result = layers.to_visible_state(ui_state)
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


def test_artist_on_visible():
    """Test case to understand Artist code

    The Artist class has two responsibilities
        - Toggle renderer.visible
        - Call viewer.render(old_state) based on visible state

    Viewers are expected to be a dict with dataset names as keys

    Renderers are expected to be a dict[str]list indexed renderers[name][figure]

    """
    from forest import db
    old_state = db.State()
    label = "label"
    renderers = [unittest.mock.Mock(), unittest.mock.Mock()]  # Per dataset per figure
    viewer = unittest.mock.Mock()  # Per dataset
    visible_state = {
        label: [True, False]
    }
    artist = layers.Artist({label: viewer}, {label: renderers})
    artist.on_state(old_state)
    artist.on_visible(layers.on_visible_state(visible_state))
    assert renderers[0].visible == True
    assert renderers[1].visible == False
    viewer.render.assert_called_once_with(old_state)
