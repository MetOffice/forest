import copy
import bokeh.models
import bokeh.layouts
import numpy as np
from typing import Iterable, List
from forest import rx
from forest.redux import Action, State, Store
from forest.observe import Observable
from forest.db.util import autolabel


SET_FIGURES = "SET_FIGURES"
LAYERS_ON_ADD = "LAYERS_ON_ADD"
LAYERS_ON_REMOVE = "LAYERS_ON_REMOVE"
LAYERS_ON_VISIBLE_STATE = "LAYERS_ON_VISIBLE_STATE"
LAYERS_ON_RADIO_BUTTON = "LAYERS_ON_RADIO_BUTTON"
LAYERS_SET_LABEL = "LAYERS_SET_LABEL"
LAYERS_SET_VISIBLE = "LAYERS_SET_VISIBLE"


def set_figures(n: int) -> Action:
    return {"kind": SET_FIGURES, "payload": n}


def on_add() -> Action:
    return {"kind": LAYERS_ON_ADD}


def on_remove() -> Action:
    return {"kind": LAYERS_ON_REMOVE}


def on_visible_state(visible_state) -> Action:
    return {"kind": LAYERS_ON_VISIBLE_STATE, "payload": visible_state}


def on_radio_button(row_index: int, active: List[int]) -> Action:
    return {
        "kind": LAYERS_ON_RADIO_BUTTON,
        "payload": {"row_index": row_index, "active": active}
    }


def set_label(index: int, label: str) -> Action:
    """Set i-th layer label"""
    return {"kind": LAYERS_SET_LABEL, "payload": {"index": index, "label": label}}


def set_visible(payload) -> Action:
    return {"kind": LAYERS_SET_VISIBLE, "payload": payload}


def middleware(store: Store, action: Action) -> Iterable[Action]:
    """Action generator given current state and action"""
    yield action


def reducer(state: State, action: Action) -> State:
    """Combine state and action to produce new state"""
    state = copy.deepcopy(state)
    kind = action["kind"]
    if kind == SET_FIGURES:
        state["figures"] = action["payload"]
    elif kind == LAYERS_ON_ADD:
        labels = state.get("layers", [])
        labels.append(None)
        state["layers"] = labels
    elif kind == LAYERS_ON_REMOVE:
        labels = state.get("layers", [])
        state["layers"] = labels[:-1]
    elif kind == LAYERS_SET_LABEL:
        index = action["payload"]["index"]
        label = action["payload"]["label"]
        labels = state.get("layers", [])

        # Pad labels with None for each missing element
        missing_elements = (index + 1) - len(labels)
        if missing_elements > 0:
            labels += missing_elements * [None]
        labels[index] = label
        state["layers"] = labels
    elif kind == LAYERS_SET_VISIBLE:
        visible = state.get("visible", {})
        visible.update(action["payload"])
        state["visible"] = visible
    return state


def to_visible_state(ui_state):
    """Maps user interface settings to visible flags

    >>> ui_state = {
        'layers': ['label'],
        'visible': [[1, 2]]
    }
    >>> to_visible_state(ui_state)
    {
        'label': [False, True, True]
    }

    """
    labels = ui_state.get("layers", [])
    active_list = ui_state.get("visible", [])
    result = {}
    for label, active in zip(labels, active_list):
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


def _connect(view, store):
    stream = (rx.Stream()
                .listen_to(store)
                .map(view.to_props)
                .filter(lambda x: x is not None)
                .distinct())
    stream.map(lambda props: view.render(*props))


class Counter:
    """Reactive layer counter component"""
    def __init__(self):
        self.div = bokeh.models.Div(text="Hello, world")
        self.layout = bokeh.layouts.row(self.div)

    def connect(self, store):
        _connect(self, store)
        return self

    @staticmethod
    def to_props(state):
        return (len(state.get("layers", [])),)

    def render(self, n):
        self.div.text = f"Rows: {n}"


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
        n = self.labels.index(new) + 1 # Select 0-indexed
        self.notify(set_figures(n))


class FigureRow:
    def __init__(self, figures):
        self.figures = figures
        self.layout = bokeh.layouts.row(*figures,
                sizing_mode="stretch_both")
        self.layout.children = [self.figures[0]]  # Trick to keep correct sizing modes

    def connect(self, store):
        stream = (rx.Stream()
                    .listen_to(store)
                    .map(self.to_props)
                    .filter(lambda x: x is not None)
                    .distinct())
        stream.map(lambda props: self.render(*props))

    def to_props(self, state):
        try:
            return (state["figures"],)
        except KeyError:
            pass

    def render(self, n):
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


class LeftCenterRight:
    # TODO: Inline this class into Controls
    def __init__(self, controls):
        self.controls = controls

    def connect(self, store):
        stream = (rx.Stream()
                    .listen_to(store)
                    .map(self.to_props)
                    .filter(lambda x: x is not None)
                    .distinct())
        stream.map(lambda props: self.render(*props))

    def to_props(self, state):
        try:
            return (state["figures"],)
        except KeyError:
            pass

    def render(self, n):
        if int(n) == 1:
            self.controls.labels = ["Show"]
        elif int(n) == 2:
            self.controls.labels = ["L", "R"]
        elif int(n) == 3:
            self.controls.labels = ["L", "C", "R"]


class Controls(Observable):
    """Collection of user interface components to manage layers"""
    def __init__(self, menu):
        self.menu = menu  # TODO: Derive this from application state
        self.defaults = {
            "label": "Model/observation",
            "flags": [False, False, False]
        }
        self.groups = []
        self.dropdowns = []
        self.buttons = {
            "add": bokeh.models.Button(label="Add", width=50),
            "remove": bokeh.models.Button(label="Remove", width=50)
        }
        self.buttons["add"].on_click(self.on_click_add)
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
        return (state.get("layers", []),)

    def render(self, labels):
        """Display latest application state in user interface

        :param n: integer representing number of rows
        """
        self.models = dict(enumerate(labels))  # TODO: remove this

        n = len(labels)
        nrows = len(self.columns["rows"].children) # - 1
        if n > nrows:
            for _ in range(n - nrows):
                self.add_row()
        if n < nrows:
            for _ in range(nrows - n):
                self.remove_row()

        # Set dropdown labels
        for label, dropdown in zip(labels, self.dropdowns):
            if label is None:
                dropdown.label = self.defaults["label"]
            else:
                dropdown.label = label

    def on_click_add(self):
        """Event-handler when Add button is clicked"""
        self.notify(on_add())

    def on_click_remove(self):
        """Event-handler when Remove button is clicked"""
        self.notify(on_remove())

    def select(self, name):
        """Select particular layers and visibility states

        .. note:: Called in main.py to select first layer
        """
        return
        self.groups[0].active = [0]
        dropdown = self.dropdowns[0]
        for k, v in dropdown.menu:
            if k == name:
                dropdown.value = v

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels
        for g in self.groups:
            g.labels = labels

    def add_row(self):
        """Add a bokeh.layouts.row with a dropdown and radiobuttongroup"""
        row_index = len(self.columns["rows"].children)

        # Dropdown
        dropdown = bokeh.models.Dropdown(
                menu=self.menu,
                label=self.defaults["label"],
                width=230,)
        dropdown.on_change('value', self.on_label(row_index))
        self.dropdowns.append(dropdown)

        # Button group
        button_group = bokeh.models.CheckboxButtonGroup(
                labels=self.labels,
                width=50)
        button_group.on_change("active", self.on_radio_button(row_index))
        self.groups.append(button_group)

        # Row
        row = bokeh.layouts.row(dropdown, button_group)
        self.columns["rows"].children.append(row)

    def remove_row(self):
        """Remove a row from user interface"""
        if len(self.columns["rows"].children) > 0:
            self.dropdowns.pop()
            self.groups.pop()
            self.columns["rows"].children.pop()

    def on_label(self, row_index: int):
        """Notify listeners of set_label action"""
        def _callback(attr, old, new):
            if old != new:
                self.notify(set_label(row_index, new))
        return _callback

    def on_radio_button(self, row_index: int):
        """Translate radio button event into Action"""
        def _callback(attr, old, new):
            self.notify(on_radio_button(row_index, new))
        return _callback


class Artist(object):
    """Applies visible and render logic to viewers and renderers


    This could easily be broken into two classes, one responsible
    for maintaining ``renderer.visible`` and one for calling
    ``viewer.render(state)``


    .. note:: This should be middleware that applies logic
              given current state and an action
    """
    def __init__(self, viewers, renderers):
        self.viewers = viewers
        self.renderers = renderers
        self.visible_state = None
        self.state = None

    def on_visible(self, action):
        """

        Uses current visible layers and incoming visible state to toggle
        on/off GlyphRenderers

        """
        # Ignore actions for now
        # TODO: Refactor to use application state or state_to_props
        kind = action["kind"]
        if kind != LAYERS_ON_VISIBLE_STATE:
            return

        visible_state = action["payload"]

        if self.visible_state is not None:
            # Hide deselected states
            lost_items = (
                    set(self.flatten(self.visible_state)) -
                    set(self.flatten(visible_state)))
            for key, i, _ in lost_items:
                self.renderers[key][i].visible = False

        # Sync visible states with menu choices
        states = set(self.flatten(visible_state))
        hidden = [(i, j) for i, j, v in states if not v]
        visible = [(i, j) for i, j, v in states if v]
        for i, j in hidden:
            self.renderers[i][j].visible = False
        for i, j in visible:
            self.renderers[i][j].visible = True

        self.visible_state = dict(visible_state)
        self.render()

    @staticmethod
    def flatten(state):
        items = []
        for key, flags in state.items():
            items += [(key, i, f) for i, f in enumerate(flags)]
        return items

    def on_state(self, app_state):
        """On application state handler"""
        # print("Artist: {}".format(app_state))
        self.state = app_state
        self.render()

    def render(self):
        """

        Notify visible viewers to render themselves given most
        recently received application state

        """
        if self.visible_state is None:
            return
        if self.state is None:
            return
        for name in self.visible_state:
            viewer = self.viewers[name]
            viewer.render(self.state)
