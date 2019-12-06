"""User interface"""
import bokeh.models
import bokeh.layouts
from forest.observe import Observable
from forest.db.util import autolabel
from forest.db.control import set_value


class DatasetUI(Observable):
    """User interface to select dataset(s)"""
    def __init__(self):
        self.dropdown = bokeh.models.Dropdown(
                label="Model/observation",
                width=350)
        autolabel(self.dropdown)
        self.dropdown.on_change("value", self.callback)
        self.layout = bokeh.layouts.column(self.dropdown)
        super().__init__()

    def callback(self, attr, old, new):
        self.notify(set_value("label", new))

    def render(self, state):
        """Configure dropdown menus"""
        assert isinstance(state, dict), "Only support dict"
        patterns = state.get("patterns", [])
        self.dropdown.menu = patterns
        self.dropdown.disabled = len(patterns) == 0
        if ("pattern" in state) and ("patterns" in state):
            for _, pattern in state["patterns"]:
                if pattern == state["pattern"]:
                    self.dropdown.value = pattern
