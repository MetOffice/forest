import forest.components


def test_render():
    modal = forest.components.Modal()
    state = {
        "patterns": [("A", None), ("B", None)]
    }
    modal.view.render(state)
    assert modal.view.selects["dataset"].options == ["A", "B"]


def test_render_edit_mode():
    modal = forest.components.Modal()
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
    modal.view.render(state)
    assert modal.view.inputs["name"].value == "Label-5"
