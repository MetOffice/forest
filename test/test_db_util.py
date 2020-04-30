import bokeh.models
import forest.db.util
from unittest.mock import Mock, sentinel


def test_autolabel():
    label, value = "Label", "value"
    event = Mock()
    event.item = value
    dropdown = bokeh.models.Dropdown(menu=[(label, value)])
    callback = forest.db.util.autolabel(dropdown)
    callback(event)
    assert dropdown.label == label
