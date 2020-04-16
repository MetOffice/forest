import forest.components


def test_render():
    view = forest.components.Modal()
    state = {
        "patterns": [("A", None), ("B", None)]
    }
    view.render(state)
    assert view.selects["dataset"].options == ["A", "B"]


def test_render_edit_mode():
    view = forest.components.Modal()
    state = {
        "layers": {
            "mode": {
                "index": 42,
                "state": "edit"
            },
            "index": {
                42: {
                    "label": "Label-5"
                }
            }
        }
    }
    view.render(state)
    assert view.inputs["name"].value == "Label-5"
