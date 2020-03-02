import bokeh.models
import forest.db.util


def test_autolabel():
    dropdown = bokeh.models.Dropdown(menu=[("label", "value")])
    callback = forest.db.util.autolabel(dropdown)
    callback(None, None, "value")
    assert dropdown.label == "label"
