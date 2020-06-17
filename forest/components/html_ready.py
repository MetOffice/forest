"""
Component to detect page-loaded event
"""
import forest.actions
import forest.state
from forest.observe import Observable


class HTMLReady(Observable):
    def __init__(self, hidden_button):
        self.hidden_button = hidden_button
        self.hidden_button.on_click(self.on_click)
        super().__init__()

    def connect(self, store):
        self.add_subscriber(store.dispatch)

    def on_click(self):
        # TODO: Remove action.to_dict() when Actions are supported
        self.notify(forest.actions.html_loaded().to_dict())


def reducer(state, action):
    """Add HTML loaded action to state"""
    if isinstance(action, dict):
        try:
            action = forest.actions.Action.from_dict(action)
        except TypeError:
            # TODO: Remove try/except when Actions are supported
            return state
    if isinstance(state, dict):
        state = forest.state.State.from_dict(state)
    if action.kind == forest.actions.HTML_LOADED:
        state.bokeh.html_loaded = True
    return state.to_dict()
