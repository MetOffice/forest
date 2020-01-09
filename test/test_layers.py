import unittest.mock
from forest import layers


def test_figure_dropdown():
    listener = unittest.mock.Mock()
    ui = layers.FigureUI()
    ui.subscribe(listener)
    ui.on_change(None, None, 1)
    listener.assert_called_once_with(layers.set_figures(1))


def test_reducer():
    state = layers.reducer({}, layers.set_figures(3))
    assert state == {"figures": 3}
