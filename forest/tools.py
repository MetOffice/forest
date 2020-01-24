"""
Tools UI Panel Components
-------------------------

Actions
~~~~~~~

Actions are tiny blobs of data used to
communicate with other parts of the system. They
help de-couple the entity creating an action from
the components that depend on state changes

.. data:: ON_TOGGLE_TOOL

    Constant to indicate toggle action

.. autofunction:: toggle_tool

"""

import copy
from forest.observe import Observable
from forest.redux import Action, State
import bokeh.layouts
import bokeh.models

ON_TOGGLE_TOOL = "TOGGLE_TOOL_VISIBILITY"


def reducer(state: State, action: Action):
    state = copy.deepcopy(state)
    if action["kind"] == ON_TOGGLE_TOOL:
        if state.get("tools") is None:
            state["tools"] = {}
        if state["tools"].get(action["tool_name"], False) == True:
            state["tools"][action["tool_name"]] = False
        else:
            state["tools"][action["tool_name"]] = True
    return state

def on_toggle_tool(tool_name) -> Action:
    return {"kind": ON_TOGGLE_TOOL, "tool_name": tool_name}

class ToolsPanel(Observable):
    """ A panel that contains buttons to turn extra tools on and off"""
    def __init__(self):
        self.buttons = {
            "toggle_time_series": bokeh.models.Button(label="Display Time Series"),
            "toggle_profile": bokeh.models.Button(label="Display Profile")}
        self.buttons["toggle_time_series"].on_click(self.on_click_time_series)
        self.buttons["toggle_profile"].on_click(self.on_click_profile)
        super().__init__()

    def connect(self, store):
        self.add_subscriber(store.dispatch)
        return self

    def on_click_time_series(self):
        """update the store."""
        self.notify(on_toggle_tool("time_series"))

    def on_click_profile(self):
        """update the store."""
        self.notify(on_toggle_tool("profile"))

class ToolLayout:
    def __init__(self, series_figure, profile_figure):
        self.figures_row = bokeh.layouts.row(
                name="series") # TODO: This name is used by CSS somewhere 
        self.series_figure = series_figure
        self.profile_figure = profile_figure

    def connect(self, store):
        store.add_subscriber(self.render)

    def render(self, state):
        self.figures_row.children = []
        for tool_name, value in state.get("tools", {}).items():
            if tool_name == "time_series" and value:
                self.figures_row.children.append(self.series_figure)
            if tool_name == "profile" and value:
                self.figures_row.children.append(self.profile_figure)


