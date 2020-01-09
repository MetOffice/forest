from forest import layers


def test_reducer():
    state = layers.reducer({}, layers.set_figures(3))
    assert state == {"figures": 3}
