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


def test_reducer():
    state = layers.reducer({}, layers.set_figures(3))
    assert state == {"figures": 3}
