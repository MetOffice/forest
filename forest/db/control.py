"""Control navigation of FOREST data"""
import copy
from functools import wraps
import bokeh.models
import bokeh.layouts
from . import util
from collections import namedtuple


__all__ = [
    "State",
    "Observable",
    "Controls",
    "next_state",
    "initial_state"
]


def export(obj):
    if obj.__name__ not in __all__:
        __all__.append(obj.__name__)
    return obj


SET_VALUE = "SET_VALUE"
NEXT_VALUE = "NEXT_VALUE"
PREVIOUS_VALUE = "PREVIOUS_VALUE"


@export
def set_value(key, value):
    return dict(kind=SET_VALUE, payload=locals())


@export
def next_value(item_key, items_key):
    return dict(kind=NEXT_VALUE, payload=locals())


@export
def previous_value(item_key, items_key):
    return dict(kind=PREVIOUS_VALUE, payload=locals())


State = namedtuple("State", (
    "pattern",
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


def next_state(current, **kwargs):
    _kwargs = current._asdict()
    _kwargs.update(kwargs)
    return State(**_kwargs)


class Observable(object):
    def __init__(self):
        self.subscribers = []

    def subscribe(self, callback):
        self.subscribers.append(callback)

    def notify(self, state):
        for callback in self.subscribers:
            callback(state)


def initial_state(database, pattern=None):
    """Find initial state given database"""
    variables = database.variables(pattern)
    variable = first(variables)
    initial_times = database.initial_times(pattern, variable)
    initial_time = None
    if len(initial_times) > 0:
        initial_time = max(initial_times)
    valid_times = database.valid_times(
        variable=variable,
        pattern=pattern,
        initial_time=initial_time)
    valid_time = None
    if len(valid_times) > 0:
        valid_time = min(valid_times)
    pressures = database.pressures(variable, pattern, initial_time)
    pressures = list(reversed(sorted(pressures)))
    pressure = None
    if len(pressures) > 0:
        pressure = pressures[0]
    state = State(
        pattern=pattern,
        variable=variable,
        variables=variables,
        initial_time=initial_time,
        initial_times=initial_times,
        valid_time=valid_time,
        valid_times=valid_times,
        pressures=pressures,
        pressure=pressure)
    return state


def first(items):
    for item in items:
        return item


@export
class Store(Observable):
    def __init__(self, reducer, initial_state=None, middlewares=None):
        self.reducer = reducer
        self.state = initial_state if initial_state is not None else {}
        if middlewares is not None:
            mws = [m(self) for m in middlewares]
            f = self.dispatch
            for mw in reversed(mws):
                f = mw(f)
            self.dispatch = f
        super().__init__()

    def dispatch(self, action):
        self.state = self.reducer(self.state, action)
        self.notify(self.state)


@export
def reducer(state, action):
    state = copy.copy(state)
    kind = action["kind"]
    if kind == SET_VALUE:
        payload = action["payload"]
        key, value = payload["key"], payload["value"]
        state[key] = value
    return state


def middleware(f):
    """Curries functions to satisfy middleware signature"""
    @wraps(f)
    def outer(*args):
        def inner(next_dispatch):
            def inner_most(action):
                f(*args, next_dispatch, action)
            return inner_most
        return inner
    return outer


@export
class Log(object):
    """Logs actions"""
    def __init__(self):
        self.actions = []

    @middleware
    def __call__(self, store, next_dispatch, action):
        value = next_dispatch(action)
        self.actions.append(action)
        return value


@export
class InverseCoordinate(object):
    """Translate actions on inverted coordinates"""
    def __init__(self, name):
        self.name = name

    @middleware
    def __call__(self, store, next_dispatch, action):
        kind = action["kind"]
        if kind in [NEXT_VALUE, PREVIOUS_VALUE]:
            if self.name == action["payload"]["item_key"]:
                return next_dispatch(self.invert(action))
        return next_dispatch(action)

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
@middleware
def next_previous(store, next_dispatch, action):
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
        return next_dispatch(set_value(item_key, value))
    return next_dispatch(action)


def button_reducer(state, action):
    items, item = get_items(state, action)
    if action.category == "pressure":
        instruction = reverse(action.instruction)
    else:
        instruction = action.instruction
    if instruction == "next":
        item = next_item(items, item)
    else:
        item = previous_item(items, item)
    return next_state(state, **{action.category: item})


def reverse(instruction):
    if instruction == "next":
        return "previous"
    else:
        return "next"


def get_items(state, action):
    if action.category == 'initial_time':
        return state.initial_times, state.initial_time
    elif action.category == 'valid_time':
        return state.valid_times, state.valid_time
    elif action.category == 'pressure':
        return state.pressures, state.pressure
    else:
        raise Exception("Unrecognised category: {}".format(action.category))


def next_item(items, item):
    if items is None:
        return None
    if item is None:
        return max(items)
    items = list(sorted(items))
    i = items.index(item)
    return items[(i + 1) % len(items)]


def previous_item(items, item):
    if items is None:
        return None
    if item is None:
        return min(items)
    items = list(sorted(items))
    i = items.index(item)
    return items[i - 1]


class Controls(Observable):
    def __init__(self, database, patterns=None, state=None):
        if patterns is None:
            patterns = []
        self.patterns = patterns
        self.database = database
        if state is None:
            state = State()
        self.state = state
        self.view = ControlView()
        self.view.subscribe(self.on_message)
        super().__init__()

    @property
    def layout(self):
        return self.view.layout

    def render(self, state):
        self.view.render(state)

    def on_message(self, message):
        state = self.modify(self.state, message)
        if state is not None:
            self.notify(state)
            self.state = state

    @middleware
    def __call__(self, store, next_dispatch, action):
        if action["kind"] == SET_VALUE:
            key = action["payload"]["key"]
            value = action["payload"]["value"]
            if key == "pressure":
                return next_dispatch(set_value(key, float(value)))
            elif key == "pattern":
                variables = self.database.variables(pattern=value)
                initial_times = self.database.initial_times(pattern=value)
                initial_times = list(reversed(initial_times))
                next_dispatch(action)
                next_dispatch(set_value("variables", variables))
                next_dispatch(set_value("initial_times", initial_times))
                return
            elif key == "initial_time":
                for attr in ["pattern", "variable"]:
                    if attr not in store.state:
                        return next_dispatch(action)
                valid_times = self.database.valid_times(
                    pattern=store.state["pattern"],
                    variable=store.state["variable"],
                    initial_time=value)
                valid_times = sorted(set(valid_times))
                next_dispatch(action)
                next_dispatch(set_value("valid_times", valid_times))
                return
        return next_dispatch(action)

    def modify(self, state, message):
        """Adjust state given message"""
        print(message)
        if message.kind == 'dropdown':
            key, value = message.payload
            if (key == 'pressure') and (value is not None):
                value = float(value)
            state = next_state(state, **{key: value})
            if key != 'pressure':
                if state.pattern is not None:
                    variables = self.database.variables(pattern=state.pattern)
                    state = next_state(state, variables=variables)

                    initial_times = list(reversed(
                        self.database.initial_times(pattern=state.pattern)))
                    state = next_state(state, initial_times=initial_times)

                if state.initial_time is not None:
                    pressures = self.database.pressures(
                        pattern=state.pattern,
                        variable=state.variable,
                        initial_time=state.initial_time)
                    pressures = list(reversed(pressures))
                    state = next_state(state, pressures=pressures)

                if state.initial_time is not None:
                    valid_times = self.database.valid_times(
                        pattern=state.pattern,
                        variable=state.variable,
                        initial_time=state.initial_time)
                    valid_times = sorted(set(valid_times))
                    state = next_state(state, valid_times=valid_times)

        if message.kind == 'button':
            state = button_reducer(state, message)
        return state


@export
class ControlView(Observable):
    def __init__(self):
        dropdown_width = 180
        button_width = 75
        self.dropdowns = {
            "pattern": bokeh.models.Dropdown(
                label="Model/observation"),
            "variable": bokeh.models.Dropdown(
                label="Variable"),
            "initial_time": bokeh.models.Dropdown(
                label="Initial time",
                width=dropdown_width),
            "valid_time": bokeh.models.Dropdown(
                label="Valid time",
                width=dropdown_width),
            "pressure": bokeh.models.Dropdown(
                label="Pressure",
                width=dropdown_width)
        }
        for key, dropdown in self.dropdowns.items():
            util.autolabel(dropdown)
            util.autowarn(dropdown)
            dropdown.on_change("value", self.on_change(key))
        self.rows = {}
        self.buttons = {}
        for key, items_key in [
                ('pressure', ''),
                ('valid_time', 'valid_times'),
                ('initial_time', 'initial_times')]:
            self.buttons[key] = {
                'next': bokeh.models.Button(
                    label="Next",
                    width=button_width),
                'previous': bokeh.models.Button(
                    label="Previous",
                    width=button_width),
            }
            self.buttons[key]['next'].on_click(
                self.on_next(key, items_key))
            self.buttons[key]['previous'].on_click(
                self.on_previous(key, items_key))
            self.rows[key] = bokeh.layouts.row(
                self.buttons[key]["previous"],
                self.dropdowns[key],
                self.buttons[key]["next"])
        self.layout = bokeh.layouts.column(
            self.dropdowns["pattern"],
            self.dropdowns["variable"],
            self.rows["initial_time"],
            self.rows["valid_time"],
            self.rows["pressure"])
        super().__init__()

    def on_change(self, key):
        def callback(attr, old, new):
            self.notify(set_value(key, new))
        return callback

    def on_next(self, item_key, items_key):
        def callback():
            self.notify(next_value(item_key, items_key))
        return callback

    def on_previous(self, item_key, items_key):
        def callback():
            self.notify(previous_value(item_key, items_key))
        return callback

    def render(self, state):
        """Configure dropdown menus"""
        for key, values in [
                ("variable", state.variables),
                ("initial_time", state.initial_times),
                ("valid_time", state.valid_times),
                ("pressure", state.pressures)]:
            if values is None:
                disabled = True
            else:
                disabled = len(values) == 0
                if key == "pressure":
                    menu = [(self.hpa(p), str(p)) for p in state.pressures]
                else:
                    menu = self.menu(values)
                self.dropdowns[key].menu = menu
            self.dropdowns[key].disabled = disabled
            if key in self.buttons:
                self.buttons[key]["next"].disabled = disabled
                self.buttons[key]["previous"].disabled = disabled

        if state.pattern is not None:
            self.dropdowns['pattern'].value = state.pattern

        if state.variable is not None:
            self.dropdowns['variable'].value = state.variable

        if state.initial_time is not None:
            self.dropdowns['initial_time'].value = state.initial_time

        if state.pressure is not None:
            self.dropdowns["pressure"].value = str(state.pressure)

        if state.valid_time is not None:
            self.dropdowns['valid_time'].value = state.valid_time

    @staticmethod
    def menu(values, formatter=str):
        return [(formatter(o), formatter(o)) for o in values]

    @staticmethod
    def hpa(p):
        if p < 1:
            return "{}hPa".format(str(p))
        return "{}hPa".format(int(p))
