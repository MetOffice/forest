import forest.components


def test_render():
    view = forest.components.Modal()
    view.render({"patterns": [("A", "*"), ("B", "*")]})
    assert view.select.options == ["A", "B"]
