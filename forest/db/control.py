"""Control navigation of FOREST data"""
import bokeh.models
import bokeh.layouts
from . import util
from collections import namedtuple


__all__ = [
    "State",
    "ButtonClick",
    "Message",
    "Observable",
    "Controls",
    "next_state"
]


State = namedtuple("State", (
    "pattern",
    "variable",
    "variables",
    "initial_time",
    "initial_times",
    "valid_time",
    "pressure",
    "pressures",
    "surface"))
State.__new__.__defaults__ = (None,) * len(State._fields)


class Message(object):
    def __init__(self, kind, payload):
        self.kind = kind
        self.payload = payload

    @classmethod
    def button(cls, category, instruction):
        return ButtonClick(category, instruction)


class ButtonClick(object):
    kind = "button"

    def __init__(self, category, value):
        self.category = category
        self.value = value

    def __repr__(self):
        if self.__class__.__module__ is not None:
            names = (self.__class__.__module__, self.__class__.__name__)
        else:
            names = (self.__class__.__name__,)

        def stringify(value):
            if isinstance(value, str):
                return "'{}'".format(value)
            else:
                return str(value)

        args = (self.category, self.value)
        return "{}({})".format(
            ".".join(names),
            ", ".join(map(stringify, args)))


def next_state(current, **kwargs):
    _kwargs = current._asdict()
    _kwargs.update(kwargs)
    return State(**_kwargs)


def next_item(items, item):
    i = items.index(item)
    return items[i + 1]


class Observable(object):
    def __init__(self):
        self.subscribers = []

    def subscribe(self, callback):
        self.subscribers.append(callback)

    def notify(self, state):
        for callback in self.subscribers:
            callback(state)


class Controls(Observable):
    def __init__(self, database, patterns=None):
        if patterns is None:
            patterns = []
        self.patterns = patterns
        self.database = database
        self.state = State()
        dropdown_width = 180
        button_width = 75
        self.dropdowns = {
            "pattern": bokeh.models.Dropdown(
                label="Model/observation",
                menu=patterns),
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
        for key in ['pressure', 'valid_time', 'initial_time']:
            self.buttons[key] = {
                'next': bokeh.models.Button(
                    label="Next",
                    width=button_width),
                'previous': bokeh.models.Button(
                    label="Previous",
                    width=button_width),
            }
            self.buttons[key]['next'].on_click(self.on_click(key, 'next'))
            self.buttons[key]['previous'].on_click(self.on_click(key, 'previous'))
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

        # Connect state changes to render
        self.subscribe(self.render)

    def render(self, state):
        """Configure dropdown menus"""
        if state.variables is not None:
            self.dropdowns["variable"].menu = self.menu(
                state.variables)

        if state.initial_times is not None:
            self.dropdowns["initial_time"].menu = self.menu(
                state.initial_times)

        if state.pressures is not None:
            menu = [(self.hpa(p), str(p)) for p in state.pressures]
            self.dropdowns["pressure"].menu = menu

        if state.pressure is not None:
            self.dropdowns["pressure"].value = str(state.pressure)

        if state.initial_time is not None:
            valid_times = self.database.valid_times(
                pattern=state.pattern,
                variable=state.variable,
                initial_time=state.initial_time)
            valid_times = sorted(set(valid_times))
            self.dropdowns["valid_time"].menu = self.menu(valid_times)

    def on_change(self, key):
        """Wire up bokeh on_change callbacks to State changes"""
        def callback(attr, old, new):
            self.send(Message("dropdown", (key, new)))
        return callback

    def on_click(self, category, value):
        def callback():
            self.send(ButtonClick(category, value))
        return callback

    def send(self, message):
        state = self.modify(self.state, message)
        self.notify(state)
        self.state = state

    def modify(self, state, message):
        """Adjust state given message"""
        if message.kind == 'dropdown':
            key, value = message.payload
            if key == 'pressure':
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
        elif message.kind == 'button':
            state = self.next_pressure(state)
        return state

    def next_pressure(self, state):
        if state.pressures is None:
            return state
        if state.pressure is None:
            return next_state(state, pressure=state.pressures[0])
        return next_state(state, pressure=self.next_item(
            state.pressures,
            state.pressure))

    @staticmethod
    def next_item(items, item):
        i = items.index(item)
        return items[i + 1]

    @staticmethod
    def menu(values, formatter=str):
        return [(formatter(o), formatter(o)) for o in values]

    @staticmethod
    def hpa(p):
        if p < 1:
            return "{}hPa".format(str(p))
        return "{}hPa".format(int(p))

    @staticmethod
    def first(items):
        try:
            return items[0]
        except IndexError:
            return
