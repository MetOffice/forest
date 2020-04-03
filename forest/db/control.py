"""Control navigation of FOREST data"""
import copy
import datetime as dt
import numpy as np
import bokeh.models
import bokeh.layouts
from . import util
from collections import namedtuple
from forest.observe import Observable
from forest.util import to_datetime as _to_datetime
from forest.export import export
from typing import List, Any


__all__ = [
    "State",
]


# Message to user when option not available
UNAVAILABLE = "Please specify"

# Action keys
SET_VALUE = "SET_VALUE"
NEXT_VALUE = "NEXT_VALUE"
PREVIOUS_VALUE = "PREVIOUS_VALUE"

@export
def set_value(key, value):
    return dict(kind=SET_VALUE, payload=locals())

@export
def next_valid_time():
    return next_value("valid_time", "valid_times")


@export
def next_initial_time():
    return next_value("initial_time", "initial_times")


@export
def next_value(item_key, items_key):
    return dict(kind=NEXT_VALUE, payload=locals())


@export
def previous_valid_time():
    return previous_value("valid_time", "valid_times")


@export
def previous_initial_time():
    return previous_value("initial_time", "initial_times")


@export
def previous_value(item_key, items_key):
    return dict(kind=PREVIOUS_VALUE, payload=locals())


State = namedtuple("State", (
    "pattern",
    "patterns",
    "variable",
    "variables",
    "initial_time",
    "initial_times",
    "valid_time",
    "valid_times",
    "pressure",
    "pressures",
    "valid_format"))
State.__new__.__defaults__ = (None,) * len(State._fields)

def statehash(self):
    return hash((self.pattern, str(self.patterns), self.variable, self.initial_time, str(self.initial_times), self.valid_time, str(self.valid_times), self.pressure, str(self.pressures), self.valid_format))

def time_equal(a, b):
    if (a is None) and (b is None):
        return True
    elif (a is None) or (b is None):
        return False
    else:
        return _to_datetime(a) == _to_datetime(b)

_vto_datetime = np.vectorize(_to_datetime)

def time_array_equal(x, y):
    if (x is None) and (y is None):
        return True
    elif (x is None) or (y is None):
        return False
    elif (len(x) == 0) or (len(y) == 0):
        return x == y
    return np.all(_vto_datetime(x) == _vto_datetime(y))

def equal_value(a, b):
    if (a is None) and (b is None):
        return True
    elif (a is None) or (b is None):
        return False
    else:
        return np.allclose(a, b)

def state_ne(self, other):
    return not (self == other)

def state_eq(self, other):
    return (
            (self.pattern == other.pattern) and
            np.all(self.patterns == other.patterns) and
            (self.variable == other.variable) and
            np.all(self.variables == other.variables) and
            time_equal(self.initial_time, other.initial_time) and
            time_array_equal(self.initial_times, other.initial_times) and
            time_equal(self.valid_time, other.valid_time) and
            time_array_equal(self.valid_times, other.valid_times) and
            equal_value(self.pressure, other.pressure) and
            np.shape(self.pressures) == np.shape(other.pressures) and
            equal_value(self.pressures, other.pressures)
    )

State.__hash__ = statehash
State.__eq__ = state_eq
State.__ne__ = state_ne


@export
def initial_state(navigator, pattern=None):
    """Find initial state given navigator"""
    state = {}
    state["pattern"] = pattern
    variables = navigator.variables(pattern)
    state["variables"] = variables
    if len(variables) == 0:
        return state
    variable = variables[0]
    state["variable"] = variable
    initial_times = navigator.initial_times(pattern, variable)
    state["initial_times"] = initial_times
    if len(initial_times) == 0:
        return state
    initial_time = max(initial_times)
    state["initial_time"] = initial_time
    valid_times = navigator.valid_times(
        variable=variable,
        pattern=pattern,
        initial_time=initial_time)
    state["valid_times"] = valid_times
    if len(valid_times) > 0:
        state["valid_time"] = min(valid_times)
    pressures = navigator.pressures(
            variable=variable,
            pattern=pattern,
            initial_time=initial_time)
    pressures = list(reversed(sorted(pressures)))
    state["pressures"] = pressures
    if len(pressures) > 0:
        state["pressure"] = pressures[0]
    return state


@export
def reducer(state, action):
    state = copy.deepcopy(state)
    kind = action["kind"]
    if kind == SET_VALUE:
        payload = action["payload"]
        key, value = payload["key"], payload["value"]
        state[key] = value
    return state


@export
class InverseCoordinate(object):
    """Translate actions on inverted coordinates"""
    def __init__(self, name):
        self.name = name

    def __call__(self, store, action):
        if self.is_next_previous(action) and self.has_name(action):
            yield self.invert(action)
        else:
            yield action

    @staticmethod
    def is_next_previous(action):
        return action["kind"] in [NEXT_VALUE, PREVIOUS_VALUE]

    def has_name(self, action):
        return self.name == action["payload"]["item_key"]

    @staticmethod
    def invert(action):
        kind = action["kind"]
        payload = action["payload"]
        item_key = payload["item_key"]
        items_key = payload["items_key"]
        if kind == NEXT_VALUE:
            return previous_value(item_key, items_key)
        else:
            return next_value(item_key, items_key)


@export
def next_previous(store, action):
    """Translate NEXT/PREVIOUS action(s) into SET action"""
    kind = action["kind"]
    if kind in [NEXT_VALUE, PREVIOUS_VALUE]:
        payload = action["payload"]
        item_key = payload["item_key"]
        items_key = payload["items_key"]
        if items_key not in store.state:
            # No further action to be taken
            return
        items = store.state[items_key]
        if item_key in store.state:
            item = store.state[item_key]
            if kind == NEXT_VALUE:
                value = next_item(items, item)
            else:
                value = previous_item(items, item)
        else:
            if kind == NEXT_VALUE:
                value = max(items)
            else:
                value = min(items)
        yield set_value(item_key, value)
    else:
        yield action


def next_item(items, item):
    items = list(sorted(items))
    i = _index(items, item)
    return items[(i + 1) % len(items)]


def previous_item(items, item):
    items = list(sorted(items))
    i = _index(items, item)
    return items[i - 1]


def _index(items: List[Any], item: Any):
    try:
        return items.index(item)
    except ValueError as e:
        # Index of first float within tolerance
        if any(np.isclose(items, item)):
            return np.isclose(items, item).argmax()
        else:
            raise e


@export
class Navigator(object):
    """Interface for navigation menu system"""
    def variables(self, pattern):
        return ['air_temperature']

    def initial_times(self, pattern):
        return ['2019-01-01 00:00:00']

    def valid_times(self, pattern, variable, initial_time):
        return ['2019-01-01 12:00:00']

    def pressures(self, pattern, variable, initial_time):
        return [750.]


@export
class Controls(object):
    def __init__(self, navigator):
        self.navigator = navigator

    def __call__(self, store, action):
        if action["kind"] == SET_VALUE:
            key = action["payload"]["key"]
            handlers = {
                "pressure": self._pressure,
                "pattern": self._pattern,
                "variable": self._variable,
                "initial_time": self._initial_time,
            }
            if key in handlers:
                yield from handlers[key](store, action)
            else:
                yield action
        else:
            yield action

    def _pressure(self, store, action):
        key = action["payload"]["key"]
        value = action["payload"]["value"]
        try:
            value = float(value)
        except ValueError:
            print("{} is not a float".format(value))
        yield set_value(key, value)

    def _pattern(self, store, action):
        pattern = action["payload"]["value"]
        variables = self.navigator.variables(pattern=pattern)
        initial_times = self.navigator.initial_times(pattern=pattern)
        initial_times = list(reversed(initial_times))
        yield action
        yield set_value("variables", variables)
        yield set_value("initial_times", initial_times)

        # Set valid_times if pattern, variable and initial_time present
        kwargs = {
            "pattern": pattern,
            "variable": store.state.get("variable"),
            "initial_time": store.state.get("initial_time"),
        }
        if all(kwargs[k] is not None for k in ["variable", "initial_time"]):
            valid_times = self.navigator.valid_times(**kwargs)
            yield set_value("valid_times", valid_times)

    def _variable(self, store, action):
        for attr in ["pattern", "initial_time"]:
            if attr not in store.state:
                yield action
                return
        pattern = store.state["pattern"]
        variable = action["payload"]["value"]
        initial_time = store.state["initial_time"]
        valid_times = self.navigator.valid_times(
            pattern=pattern,
            variable=variable,
            initial_time=initial_time)
        valid_times = sorted(set(valid_times))
        pressures = self.navigator.pressures(
            pattern=pattern,
            variable=variable,
            initial_time=initial_time)
        pressures = list(reversed(pressures))
        yield action
        yield set_value("valid_times", valid_times)
        yield set_value("pressures", pressures)
        if ("pressure" not in store.state) and len(pressures) > 0:
            yield set_value("pressure", max(pressures))

    def _initial_time(self, store, action):
        for attr in ["pattern", "variable"]:
            if attr not in store.state:
                yield action
                return
        initial_time = action["payload"]["value"]
        valid_times = self.navigator.valid_times(
            pattern=store.state["pattern"],
            variable=store.state["variable"],
            initial_time=initial_time)
        valid_times = sorted(set(valid_times))
        yield action
        yield set_value("valid_times", valid_times)


@export
class ControlView:
    """Layout of navigation controls

    A high-level view that delegates to low-level views
    which in turn perform navigation.
    """
    def __init__(self):
        self.views = {}
        self.views["dataset"] = DatasetView()
        self.views["variable"] = DimensionView("variable", "variables", next_previous=False)
        self.views["initial_time"] = DimensionView("initial_time", "initial_times")
        self.views["valid_time"] = DimensionView("valid_time", "valid_times")
        self.views["pressure"] = DimensionView("pressure", "pressures", formatter=self.hpa)
        self.layout = bokeh.layouts.column(
            self.views["dataset"].layout,
            self.views["variable"].layout,
            self.views["initial_time"].layout,
            self.views["valid_time"].layout,
            self.views["pressure"].layout,
        )
        super().__init__()

    def connect(self, store):
        """Connect views to the store"""
        self.views["dataset"].connect(store)
        self.views["variable"].connect(store)
        self.views["initial_time"].connect(store)
        self.views["valid_time"].connect(store)
        self.views["pressure"].connect(store)

    @staticmethod
    def hpa(p):
        return format_hpa(p)


def format_hpa(p):
    """Text representation of atmospheric pressure"""
    if p is None:
        return "Pressure"
    if float(p) < 1:
        return "{}hPa".format(str(p))
    return "{}hPa".format(int(p))


class DatasetView(Observable):
    """View to select datasets

    .. note:: Currently 'pattern' is the primary key for
              dataset selection
    """
    def __init__(self):
        self._table = {}
        self.item_key = "pattern"
        self.items_key = "patterns"
        self.select = bokeh.models.Select(width=350)
        self.select.on_change("value", self.on_select)
        self.layout = bokeh.layouts.row(
            self.select
        )
        super().__init__()

    def on_select(self, attr, old, new):
        """On click handler for select widget"""
        if new == UNAVAILABLE:
            return
        value = self._table.get(new, new)
        self.notify(set_value(self.item_key, value))

    def connect(self, store):
        """Wire up component to the Store"""
        self.add_subscriber(store.dispatch)
        store.add_subscriber(self.render)

    def render(self, state):
        """Render application state"""
        pattern = state.get(self.item_key)
        patterns = state.get(self.items_key, [])
        self._table.update(patterns)
        option = self.find_label(patterns, pattern)
        options = [label for label, _ in patterns]
        self.select.options = [UNAVAILABLE] + options
        if option in options:
            self.select.value = option
        else:
            self.select.value = UNAVAILABLE

    @staticmethod
    def find_label(patterns, pattern):
        for label, _pattern in patterns:
            if _pattern == pattern:
                return label


class DimensionView(Observable):
    """Widgets used to navigate a dimension"""
    def __init__(self, item_key, items_key, next_previous=True, formatter=str):
        self.item_key = item_key
        self.items_key = items_key
        self.formatter = formatter
        self._lookup = {}  # Look-up table to convert from label to value
        self.next_previous = next_previous
        if self.next_previous:
            # Include next/previous buttons
            self.select = bokeh.models.Select(
                width=180)
            self.select.on_change("value", self.on_select)
            self.buttons = {
                "next": bokeh.models.Button(
                    label="Next",
                    width=75),
                "previous": bokeh.models.Button(
                    label="Previous",
                    width=75),
            }
            self.buttons["next"].on_click(self.on_next)
            self.buttons["previous"].on_click(self.on_previous)
            self.layout = bokeh.layouts.row(
                self.buttons["previous"],
                self.select,
                self.buttons["next"],
            )
        else:
            # Without next/previous buttons
            self.select = bokeh.models.Select(width=350)
            self.select.on_change("value", self.on_select)
            self.layout = bokeh.layouts.row(
                self.select
            )
        super().__init__()

    def on_select(self, attr, old, new):
        """Handler for select widget"""
        if new == UNAVAILABLE:
            return
        value = self._lookup.get(new, new)
        self.notify(set_value(self.item_key, value))

    def on_next(self):
        """Handler for next button"""
        self.notify(next_value(self.item_key, self.items_key))

    def on_previous(self):
        """Handler for previous button"""
        self.notify(previous_value(self.item_key, self.items_key))

    def connect(self, store):
        """Connect user interactions to the store"""
        self.add_subscriber(store.dispatch)
        store.add_subscriber(self.render)

    def render(self, state):
        """Apply state to widgets"""
        value = state.get(self.item_key)
        values = state.get(self.items_key, [])
        option = self.formatter(value)
        options = [self.formatter(value) for value in values]
        self._lookup.update(zip(options, values))
        self.select.options = [UNAVAILABLE] + options
        if option in options:
            self.select.value = option
        else:
            self.select.value = UNAVAILABLE

        # Deactivate widgets if no options available
        disabled = len(options) == 0
        self.select.disabled = disabled
        if self.next_previous:
            self.buttons["next"].disabled = disabled
            self.buttons["previous"].disabled = disabled
