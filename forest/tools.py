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
import bokeh.layouts
import bokeh.models
from forest.observe import Observable
from forest.redux import Action, State

ON_TOGGLE_TOOL = "TOGGLE_TOOL_VISIBILITY"


def reducer(state: State, action: Action):
    """ Reduce a change in state caused by the ToolsPanel"""
    state = copy.deepcopy(state)
    if action["kind"] == ON_TOGGLE_TOOL:
        if state.get("tools") is None:
            state["tools"] = {}
        state["tools"][action["tool_name"]] = action["value"]
    return state

def on_toggle_tool(tool_name, value) -> Action:
    """ Convert some arguments into an Action message to the state"""
    return {"kind": ON_TOGGLE_TOOL, "tool_name": tool_name, "value": value}

class ToolsPanel(Observable):
    """ A panel that contains buttons to turn extra tools on and off"""
    def __init__(self, available_features):

        self.buttons = {}
        for tool_name, display_name in available_features.items(): 
            self.buttons[tool_name] = bokeh.models.Toggle(label=display_name)
            self.buttons[tool_name].on_click(self.on_click(tool_name))

        self.layout = bokeh.layouts.column(*self.buttons.values())
        super().__init__()

    def connect(self, store):
        self.add_subscriber(store.dispatch)
        return self

    def on_click(self, toggle_name):
        """update the store callback."""
        def callback(toggle_state):
            self.notify(on_toggle_tool(toggle_name, toggle_state))

        return callback

class ToolLayout:
    """ Manage the row containing the tool plots """
    def __init__(self, series_figure=None, profile_figure=None):
        self.layout = bokeh.layouts.column()
        self.series_figure = series_figure
        self.profile_figure = profile_figure

    def connect(self, store):
        store.add_subscriber(self.render)

    def render(self, state):
        children = []
        for tool_name, value in state.get("tools", {}).items():
            if tool_name == "time_series" and value:
                children.append(self.series_figure)
            if tool_name == "profile" and value:
                children.append(self.profile_figure)
        self.layout.children = children
