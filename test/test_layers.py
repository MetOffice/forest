import pytest
import unittest.mock
from forest import layers

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
    ({}, layers.on_add(), {"layers": 1}),
    ({}, [layers.on_add(), layers.on_add()], {"layers": 2}),
    ({}, layers.on_remove(), {"layers": 0}),
    ({"layers": 2}, layers.on_remove(), {"layers": 1}),
])
def test_reducer(state, actions, expect):
    if isinstance(actions, dict):
        actions = [actions]
    for action in actions:
        state = layers.reducer(state, action)
    assert state == expect


def test_controls_render(listener):
    controls = layers.Controls([])
    controls.subscribe(listener)
    controls.render()
    listener.assert_called_once_with({})


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
    renderer = unittest.mock.Mock()
    viewer = unittest.mock.Mock()
    visible_state = {
        label: [True]
    }
    artist = layers.Artist({label: viewer}, {label: [renderer]})
    artist.on_state(old_state)
    artist.on_visible(visible_state)
    assert renderer.visible == True
    viewer.render.assert_called_once_with(old_state)
