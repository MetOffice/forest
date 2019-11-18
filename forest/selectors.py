"""Indirect access to State properties"""


def pressure(state):
    if isinstance(state, tuple):
        return state.pressure
    return state.get("pressure", None)
