"""
Component to detect page-loaded event
"""
import forest.actions
import forest.state


def reducer(state, action):
    """Add HTML loaded action to state"""
    if isinstance(action, dict):
        try:
            action = forest.actions.Action.from_dict(action)
        except TypeError:
            # TODO: Remove when Action is fully supported
            return state
    if isinstance(state, dict):
        state = forest.state.State.from_dict(state)
    if action.kind == forest.actions.HTML_LOADED:
        state.bokeh.html_loaded = True
    return state.to_dict()
