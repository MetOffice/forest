"""Dimension information relating to datasets"""
import copy


SET_VARIABLES = "DIMENSION_SET_VARIABLES"


def set_variables(label, values):
    """Action to set variables related to dataset"""
    return {"kind": SET_VARIABLES,
            "payload": {"label": label, "values": values}}


def reducer(state, action):
    """Encode dimension information into state"""
    state = copy.deepcopy(state)
    kind = action["kind"]
    if kind == SET_VARIABLES:
        node = state
        for key in ("dimension", action["payload"]["label"]):
            node[key] = node.get(key, {})
            node = node[key]
        node["variables"] = action["payload"]["values"]
    return state
