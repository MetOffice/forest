"""Reducer"""
import copy
from forest import (
    actions,
    redux,
    db,
    layers,
    screen,
    tools,
    colors,
    presets,
    dimension)
from forest.components import (
    html_ready,
    tiles)


def state_reducer(state, action):
    """Set State"""
    if isinstance(action, dict):
        try:
            action = actions.Action.from_dict(action)
        except TypeError:
            # TODO: Support Action throughout codebase
            return state
    # Reduce state
    state = copy.deepcopy(state)
    if action.kind == actions.SET_STATE:
        state = copy.deepcopy(action.payload)
    elif action.kind == actions.UPDATE_STATE:
        state.update(action.payload)
    return state


reducer = redux.combine_reducers(
            db.reducer,
            layers.reducer,
            screen.reducer,
            tools.reducer,
            colors.reducer,
            colors.limits_reducer,
            presets.reducer,
            tiles.reducer,
            dimension.reducer,
            html_ready.reducer,
            state_reducer)
