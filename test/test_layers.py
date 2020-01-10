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
    """Test case to understand Controls.render()

    Controls has two dicts self.models and self.flags which
    are populated from dropdown, radio button and add/remove events

    """
    controls = layers.Controls([])
    controls.subscribe(listener)
    controls.render()
    listener.assert_called_once_with({})


def test_controls_on_dropdown(listener):
    controls = layers.Controls([])
    controls.subscribe(listener)
    controls.on_dropdown(0)(None, None, "new")  # attr, old, new
    controls.on_radio(0)(None, [], [0])  # attr, old, new
    assert controls.models == {0: "new"}
    assert controls.flags == {0: [True, False, False]}
    listener.assert_has_calls([
        unittest.mock.call({}),
        unittest.mock.call({"new": [True, False, False]}),
    ])


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
    artist.on_visible(visible_state)
    assert renderers[0].visible == True
    assert renderers[1].visible == False
    viewer.render.assert_called_once_with(old_state)
