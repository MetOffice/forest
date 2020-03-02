"""
Layers
------

Users can select combinations of visualisations displayed across
multiple figures. Components for organising view layers associated
with each figure can be found here.

"""
import copy
import bokeh.models
import bokeh.layouts
import numpy as np
from collections import defaultdict
from typing import Iterable, List
from forest import rx
from forest.redux import Action, State, Store
from forest.observe import Observable
from forest.db.util import autolabel


ADD_LAYER = "LAYERS_ADD_LAYER"
ON_REMOVE = "LAYERS_ON_REMOVE"
ON_DROPDOWN = "LAYERS_ON_DROPDOWN"
ON_BUTTON_GROUP = "LAYERS_ON_BUTTON_GROUP"
SET_FIGURES = "LAYERS_SET_FIGURES"
SET_ACTIVE = "LAYERS_SET_ACTIVE"
SET_LABEL = "LAYERS_SET_LABEL"


def set_figures(n: int) -> Action:
    return {"kind": SET_FIGURES, "payload": n}


def add_layer(name) -> Action:
    return {"kind": ADD_LAYER, "payload": name}


def on_remove() -> Action:
    return {"kind": ON_REMOVE}


def on_button_group(row_index: int, active: List[int]) -> Action:
    return {
        "kind": ON_BUTTON_GROUP,
        "payload": {"row_index": row_index, "active": active}
    }


def set_active(row_index: int, active: List[int]) -> Action:
    return {
        "kind": SET_ACTIVE,
        "payload": {"row_index": row_index, "active": active}
    }


def on_dropdown(row_index: int, label: str) -> Action:
    return {
        "kind": ON_DROPDOWN,
        "payload": {"row_index": row_index, "label": label}
    }


def set_label(index: int, label: str) -> Action:
    """Set i-th layer label"""
    return {"kind": SET_LABEL, "payload": {"index": index, "label": label}}


def middleware(store: Store, action: Action) -> Iterable[Action]:
    """Action generator given current state and action"""
    kind = action["kind"]
    if kind == ON_BUTTON_GROUP:
        payload = action["payload"]
        yield set_active(payload["row_index"], payload["active"])
    elif kind == ON_DROPDOWN:
        payload = action["payload"]
        yield set_label(payload["row_index"], payload["label"])
    else:
        yield action


def reducer(state: State, action: Action) -> State:
    """Combine state and action to produce new state"""
    state = copy.deepcopy(state)
    kind = action["kind"]
    if kind in [
            ADD_LAYER,
            SET_ACTIVE,
            SET_LABEL,
            SET_FIGURES,
            ON_REMOVE]:
        layers = state.get("layers", {})
        state["layers"] = _layers_reducer(layers, action)
    return state


def _layers_reducer(state, action):
    kind = action["kind"]
    if kind == SET_FIGURES:
        state["figures"] = action["payload"]

    elif kind == ADD_LAYER:
        labels = state.get("labels", [])
        labels.append(action["payload"])
        state["labels"] = labels

    elif kind == ON_REMOVE:
        labels = state.get("labels", [])
        state["labels"] = labels[:-1]

    elif kind == SET_LABEL:
        row_index = action["payload"]["index"]
        label = action["payload"]["label"]
        labels = state.get("labels", [])

        # Pad with None for each missing element
        labels += (row_index + 1 - len(labels)) * [None]
        labels[row_index] = label
        state["labels"] = labels

    elif kind == SET_ACTIVE:
        payload = action["payload"]
        row_index = payload["row_index"]
        active = payload["active"]
        items = state.get("active", [])
        items += (row_index + 1 - len(items)) * [None]
        items[row_index] = active
        state["active"] = items

    return state


def _connect(view, store):
    stream = (rx.Stream()
                .listen_to(store)
                .map(view.to_props)
                .filter(lambda x: x is not None)
                .distinct())
    stream.map(lambda props: view.render(*props))


class FigureUI(Observable):
    """Controls how many figures are currently displayed"""
    def __init__(self):
        self.labels = [
            "Single figure",
            "Side by side",
            "3 way comparison"]
        self.select = bokeh.models.Select(
            options=self.labels,
            value="Single figure",
            width=350,
        )
        self.select.on_change("value", self.on_change)
        self.layout = bokeh.layouts.column(
            self.select,
        )
        super().__init__()

    def on_change(self, attr, old, new):
        """Emit action to set number of figures in state"""
        n = self.labels.index(new) + 1 # Select 0-indexed
        self.notify(set_figures(n))


class FigureRow:
    """Component to toggle number of displayed figures"""
    def __init__(self, figures):
        self.figures = figures
        self.layout = bokeh.layouts.row(*figures,
                sizing_mode="stretch_both",
                name="figures")
        self.layout.children = [self.figures[0]]  # Trick to keep correct sizing modes

    def connect(self, store):
        """Connect to the Store"""
        stream = (rx.Stream()
                    .listen_to(store)
                    .map(self.to_props)
                    .filter(lambda x: x is not None)
                    .distinct())
        stream.map(lambda props: self.render(*props))

    def to_props(self, state: State):
        """Select number of figures from state"""
        layers = state.get("layers", {})
        try:
            return (layers["figures"],)
        except KeyError:
            pass

    def render(self, n: int):
        """Assign figures to row"""
        if int(n) == 1:
            self.layout.children = [
                    self.figures[0]]
        elif int(n) == 2:
            self.layout.children = [
                    self.figures[0],
                    self.figures[1]]
        elif int(n) == 3:
            self.layout.children = [
                    self.figures[0],
                    self.figures[1],
                    self.figures[2]]


class LayersUI(Observable):
    """Collection of user interface components to manage layers"""
    def __init__(self, menu):
        self.menu = menu  # TODO: Derive this from application state
        self.defaults = {
            "label": "Model/observation",
            "flags": [False, False, False],
            "figure": {
                1: ["Show"],
                2: ["L", "R"],
                3: ["L", "C", "R"]
            }
        }
        self.button_groups = []
        self.dropdowns = []
        self.buttons = {
            "add": bokeh.models.Button(label="Add", width=50),
            "remove": bokeh.models.Button(label="Remove", width=50)
        }
        custom_js = bokeh.models.CustomJS(code="""
            let el = document.getElementById("modal");
            el.style.visibility = "visible";
        """)
        self.buttons["add"].js_on_click(custom_js)
        self.buttons["remove"].on_click(self.on_click_remove)
        self.columns = {
            "rows": bokeh.layouts.column(),
            "buttons": bokeh.layouts.column(
                bokeh.layouts.row(self.buttons["add"], self.buttons["remove"])
            )
        }
        self.layout = bokeh.layouts.column(
            self.columns["rows"],
            self.columns["buttons"]
        )
        self._labels = ["Show"]
        super().__init__()

    def connect(self, store):
        """Connect component to store"""
        _connect(self, store)
        return self

    @staticmethod
    def to_props(state) -> tuple:
        """Select data from state that satisfies self.render(*props)"""
        layers = state.get("layers", {})
        return (
            layers.get("labels", []),
            layers.get("active", []),
            layers.get("figures", None)
        )

    def render(self, labels, active_list, figure_index):
        """Display latest application state in user interface

        :param n: integer representing number of rows
        """
        # Match rows to number of labels
        n = len(labels)
        nrows = len(self.columns["rows"].children) # - 1
        if n > nrows:
            for label in labels[nrows:]:
                self.add_row(label)
        if n < nrows:
            for _ in range(nrows - n):
                self.remove_row()

        # Set button group active
        for button_group, active in zip(self.button_groups, active_list):
            button_group.active = active

        # Set button group labels
        if figure_index is not None:
            self.labels = self.defaults["figure"][figure_index]

    def on_click_remove(self):
        """Event-handler when Remove button is clicked"""
        self.notify(on_remove())

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels
        for g in self.button_groups:
            g.labels = labels

    def add_row(self, label=None):
        """Add a bokeh.layouts.row with a dropdown and checkboxbuttongroup"""
        row_index = len(self.columns["rows"].children)

        # Dropdown
        if label is None:
            label = self.defaults["label"]
        dropdown = bokeh.models.Dropdown(
                menu=self.menu,
                label=label,
                width=230,)
        dropdown.on_change('value', self.on_dropdown(row_index))
        self.dropdowns.append(dropdown)

        # Button group
        button_group = bokeh.models.CheckboxButtonGroup(
                labels=self.labels,
                width=50)
        button_group.on_change("active", self.on_button_group(row_index))
        self.button_groups.append(button_group)

        # Row
        row = bokeh.layouts.row(dropdown, button_group)
        self.columns["rows"].children.append(row)

    def remove_row(self):
        """Remove a row from user interface"""
        if len(self.columns["rows"].children) > 0:
            self.dropdowns.pop()
            self.button_groups.pop()
            self.columns["rows"].children.pop()

    def on_dropdown(self, row_index: int):
        """Translate event into Action"""
        def _callback(attr, old, new):
            if old != new:
                self.notify(on_dropdown(row_index, new))
        return _callback

    def on_button_group(self, row_index: int):
        """Translate event into Action"""
        def _callback(attr, old, new):
            # Note: bokeh.core.PropertyList can not be deep copied
            #       it RuntimeErrors, cast as list instead
            active = list(new)
            self.notify(on_button_group(row_index, active))
        return _callback


class ViewerConnector:
    """Connect Views to Store"""
    def __init__(self):
        self.subscribers = defaultdict(list)

    def add_label_subscriber(self, label, callback):
        """Register views that depend on a label"""
        self.subscribers[label].append(callback)

    def connect(self, store):
        """Subscribe to all store dispatch events"""
        store.add_subscriber(self.render)
        return self

    def render(self, state):
        """Notify listeners related to labels"""
        for label in self.labels(state):
            if label is None:
                continue
            for method in self.subscribers[label]:
                method(state)

    def labels(self, state):
        return state.get("layers", {}).get("labels", [])


class Artist:
    """Applies renderer.visible logic to renderers"""
    def __init__(self, renderers: dict):
        self.renderers = renderers
        self.previous_visible_state = None

    def connect(self, store):
        """Connect component to the store"""
        store.add_subscriber(self.render)
        return self

    def render(self, state: dict):
        """Update visible settings of renderers"""
        labels = state.get("layers", {}).get("labels", [])
        active_list = state.get("layers", {}).get("active", [])
        visible_state = to_visible_state(labels, active_list)
        if self.previous_visible_state is None:
            changes = diff_visible_states({}, visible_state)
        else:
            changes = diff_visible_states(
                self.previous_visible_state, visible_state)
        for label, figure_index, flag in changes:
            self.renderers[label][figure_index].visible = flag
        self.previous_visible_state = visible_state


def to_visible_state(labels: List[str], active_list: List[List[int]]) -> dict:
    """Maps user interface settings to visible flags

    >>> to_visible_state(['label'], [[1, 2]])
    {
        'label': [False, True, True]
    }

    """
    result = {}
    for label, active in zip(labels, active_list):
        if label is None:
            continue
        if label not in result:
            result[label] = [False, False, False]
        for i in active:
            result[label][i] = True
    return result


def diff_visible_states(left: dict, right: dict) -> List[tuple]:
    """Calculate changes needed to map from left to right state

    There are essentially three situations:
       1. A label has been added, apply all flags
       2. A label has been removed, mute True flags
       3. The flags of an existing label have been altered

    >>> left = {
       'label': [True, False, False]
    }
    >>> right = {
       'label': [True, True, False]
    }
    >>> diff_visible(left, right)
    [('label', 1, True)]

    A change is defined as a tuple ``(label, figure_index, flag)``, to
    make it easy to update renderer visible attributes

    .. note:: If nothing has changed an empty list is returned

    :returns: list of changes
    """
    diff = []

    # Detect additions to state
    for key in right:
        if key not in left:
            for i, flag in enumerate(right[key]):
                diff.append((key, i, flag))

    # Detect removals from state
    for key in left:
        if key not in right:
            for i, flag in enumerate(left[key]):
                if flag:
                    diff.append((key, i, False))  # Turn off True flags

    # Detect alterations to existing labels
    for key in left:
        if key in right:
            for i, flag in enumerate(left[key]):
                if flag != right[key][i]:
                    diff.append((key, i, right[key][i]))
    return diff
