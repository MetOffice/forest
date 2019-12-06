"""User interface"""
import bokeh.models
import bokeh.layouts
from forest.observe import Observable
from forest.db.util import autolabel


SET_LABEL = "SET_LABEL"


def set_label(label):
    return {"kind": SET_LABEL, "payload": {"label": label}}


def set_labels(labels):
    return {"kind": SET_LABEL, "payload": {"labels": labels}}


def reducer(state, action):
    kind = action["kind"]
    if kind == SET_LABEL:
        state.update(action["payload"])
    return state


class DatasetUI(Observable):
    """User interface to select dataset(s)"""
    def __init__(self):
        self.dropdown = bokeh.models.Dropdown(
                label="Dataset",
                width=350)
        autolabel(self.dropdown)
        self.dropdown.on_change("value", self.callback)
        self.layout = bokeh.layouts.column(self.dropdown)
        super().__init__()

    def callback(self, attr, old, new):
        self.notify(set_label(new))

    def render(self, state):
        """Configure dropdown menus"""
        assert isinstance(state, dict), "Only support dict"
        if "label" in state:
            self.dropdown.label = state["label"]
        if "labels" in state:
            self.dropdown.menu = [(str(l), str(l)) for l in state["labels"]]
