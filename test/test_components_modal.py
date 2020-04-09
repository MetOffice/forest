import forest.components


def test_render():
    view = forest.components.Modal()
    state = {
        "patterns": [("A", None), ("B", None)]
    }
    view.render(state)
    assert view.selects["dataset"].options == ["A", "B"]
