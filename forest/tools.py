"""
Tools UI Panel Components
-------------------------


"""


from forest.observe import Observable
from forest.redux import Action
import bokeh.layouts
import bokeh.models

ON_TOGGLE = "TOGGLE_VISIBILITY"


def on_toggle() -> Action:
    return {"kind": ON_TOGGLE}

class ToolsPanel(Observable):
    """ A panel that contains buttons to turn extra tools on and off"""
    def __init__(self):
        self.buttons = {"toggle_time_series": bokeh.models.Button(label="Display Time Series")}
        self.buttons["toggle_time_series"].on_click(self.on_click_time_series)
        super().__init__()

    def connect(self, store):
        self.add_subscriber(store.dispatch)
        return self

    def on_click_time_series(self):
        """update the store."""
        self.notify(on_toggle())


class ToolLayout:
    def __init__(self, tool_figure):
        self.figures_row = bokeh.layouts.row(
                tool_figure,
                name="series")
        self.tool_figure = tool_figure

    def connect(self, store):
        store.add_subscriber(self.render)

    def render(self, state):
        if state.get("time_series_visible"):
            self.figures_row.children = [self.tool_figure]
        else:
            self.figures_row.children = []


