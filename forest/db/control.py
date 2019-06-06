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
    "valid_times",
    "pressure",
    "pressures"))
State.__new__.__defaults__ = (None,) * len(State._fields)


def next_state(current, **kwargs):
    _kwargs = current._asdict()
    _kwargs.update(kwargs)
    return State(**_kwargs)


class Message(object):
    def __init__(self, kind, payload):
        self.kind = kind
        self.payload = payload

    @classmethod
    def button(cls, category, instruction):
        return ButtonClick(category, instruction)

    @classmethod
    def dropdown(cls, key, value):
        return cls("dropdown", (key, value))

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

        args = (self.kind, self.payload)
        return "{}({})".format(
            ".".join(names),
            ", ".join(map(stringify, args)))


class ButtonClick(object):
    kind = "button"

    def __init__(self, category, instruction):
        self.category = category
        self.instruction = instruction

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

        args = (self.category, self.instruction)
        return "{}({})".format(
            ".".join(names),
            ", ".join(map(stringify, args)))


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

        if state.initial_time is not None:
            self.dropdowns['initial_time'].value = state.initial_time

        if state.pressure is not None:
            self.dropdowns["pressure"].value = str(state.pressure)

        if state.valid_time is not None:
            self.dropdowns['valid_time'].value = state.valid_time

    def on_change(self, key):
        """Wire up bokeh on_change callbacks to State changes"""
        def callback(attr, old, new):
            self.send(Message("dropdown", (key, new)))
        return callback

    def on_click(self, category, instruction):
        def callback():
            self.send(ButtonClick(category, instruction))
        return callback

    def send(self, message):
        state = self.modify(self.state, message)
        if state is not None:
            self.notify(state)
            self.state = state

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
            state = self.modify_button(state, message)
        return state

    def modify_button(self, state, message):
        if message.category == 'initial_time':
            values = state.initial_times
            value = state.initial_time
        elif message.category == 'valid_time':
            values = state.valid_times
            value = state.valid_time
        elif message.category == 'pressure':
            values = state.pressures
            value = state.pressure
        if values is None:
            return state
        if message.instruction == 'previous':
            if value is None:
                value = values[0]
            else:
                i = values.index(value)
                value = values[i - 1]
        else:
            if value is None:
                value = values[0]
            else:
                i = values.index(value)
                value = values[i + 1]
        return next_state(state, **{message.category: value})

    @staticmethod
    def menu(values, formatter=str):
        return [(formatter(o), formatter(o)) for o in values]

    @staticmethod
    def hpa(p):
        if p < 1:
            return "{}hPa".format(str(p))
        return "{}hPa".format(int(p))
