"""User interface"""
import numpy as np
import datetime as dt
import bokeh.models
import bokeh.layouts


SET_DIMENSIONS = "SET_DIMENSIONS"
SET_COORDINATE = "SET_COORDINATE"
SET_SELECTED = "SET_SELECTED"


def set_dimensions(label, values):
    return dict(kind=SET_DIMENSIONS, payload=locals())


def set_selected(label):
    return dict(kind=SET_SELECTED, payload=locals())


def set_coordinate(label, name, values):
    return dict(kind=SET_COORDINATE, payload=locals())


def reducer(state, action):
    kind = action["kind"]
    if kind == SET_DIMENSIONS:
        payload = action["payload"]
        label = payload["label"]
        values = payload["values"]

        # Update groups
        gid, groups = insert_group(state.get("groups", {}), label)
        group = groups.get(gid, {})
        group["label"] = label
        groups[gid] = group

        # Update dimensions
        dimensions = state.get("dimensions", {})
        did = dimensions_id(dimensions, values)
        if did is None:
            did = next_id(dimensions)
        dimensions[did] = values

        # Link group to dimension
        group["dimensions"] = did

        state["groups"] = groups
        state["dimensions"] = dimensions
    elif kind == SET_SELECTED:
        payload = action["payload"]
        label = payload["label"]
        gid, groups = insert_group(state.get("groups", {}), label)
        state["groups"] = groups
        state["selected"] = gid
    elif kind == SET_COORDINATE:
        payload = action["payload"]
        label = payload["label"]
        name = payload["name"]
        values = payload["values"]
        gid, groups = insert_group(state.get("groups", {}), label)
        group = groups[gid]
        if "coordinates" not in group:
            group["coordinates"] = {}
        group["coordinates"].update({name: [sanitize(value) for value in values]})
        state["groups"] = groups
    return state


def sanitize(value):
    if isinstance(value, np.datetime64):
        value = value.astype(dt.datetime)
    if isinstance(value, (int, float)):
        return value
    return str(value)


def insert_group(groups, label):
    gid = label_id(groups, label)
    if gid is None:
        gid = next_id(groups)
    if gid in groups:
        groups[gid]["label"] = label
    else:
        groups[gid] = {"label": label}
    return gid, groups


def label_id(table, label):
    for k, v in table.items():
        if v.get("label", None) == label:
            return k


def dimensions_id(table, dims):
    for k, v in table.items():
        if tuple(v) == tuple(dims):
            return k


def next_id(table):
    try:
        return max(table.keys()) + 1
    except ValueError:
        return 0


class Query(object):
    """Helper class to traverse application state"""
    def __init__(self, state):
        self.state = state

    @property
    def selected_label(self):
        state = self.state
        if "selected" not in state:
            return
        if "groups" not in state:
            return
        return state["groups"][state["selected"]]["label"]

    @property
    def labels(self):
        state = self.state
        if "groups" not in state:
            return []
        return [group['label'] for _, group in sorted(state['groups'].items())]

    def dimensions(self, label):
        state = self.state
        if "dimensions" not in state:
            return
        if "groups" not in state:
            return
        dimensions = state["dimensions"]
        groups = state["groups"]
        for group in groups.values():
            if group["label"] == label:
                return dimensions[group["dimensions"]]


def find_dimensions(state, label):
    return Query(state).dimensions(label)


class Controls(object):
    def __init__(self):
        self.rows = {
            "title": bokeh.layouts.row(
                    bokeh.models.Div(text="Navigation:")),
            "variable": bokeh.layouts.row(
                    bokeh.models.Button(label="Previous"),
                    bokeh.models.Dropdown(label="Variable"),
                    bokeh.models.Button(label="Next")),
            "initial_time": bokeh.layouts.row(
                    bokeh.models.Button(label="Previous"),
                    bokeh.models.Dropdown(label="Initial time"),
                    bokeh.models.Button(label="Next")),
            "valid_time": bokeh.layouts.row(
                    bokeh.models.Button(label="Previous"),
                    bokeh.models.Dropdown(label="Valid time"),
                    bokeh.models.Button(label="Next")),
            "pressure": bokeh.layouts.row(
                    bokeh.models.Button(label="Previous"),
                    bokeh.models.Dropdown(label="Pressure"),
                    bokeh.models.Button(label="Next"))
        }
        self.layout = bokeh.layouts.column(self.rows["title"])

    def render(self, state):
        query = Query(state)
        label = query.selected_label
        dimensions = query.dimensions(label)
        children = [
            self.rows["title"]
        ]
        for d in dimensions:
            children.append(self.rows[d])
        self.layout.children = children
