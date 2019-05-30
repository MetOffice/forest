"""Control navigation of FOREST data"""
import bokeh.models
import bokeh.layouts
import _util as util
from collections import namedtuple


State = namedtuple("State", (
    "pattern",
    "variable",
    "initial_time",
    "valid_time",
    "pressure"))
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


class Controls(Observable):
    def __init__(self, database, patterns=None):
        if patterns is None:
            patterns = []
        self.patterns = patterns
        self.database = database
        self.state = State()
        self.dropdowns = {
            "pattern": bokeh.models.Dropdown(
                label="Model/observation",
                menu=patterns),
            "variable": bokeh.models.Dropdown(
                label="Variable"),
            "initial_time": bokeh.models.Dropdown(
                label="Initial time"),
            "valid_time": bokeh.models.Dropdown(
                label="Valid time"),
            "pressure": bokeh.models.Dropdown(
                label="Pressure")
        }
        for key, dropdown in self.dropdowns.items():
            util.autolabel(dropdown)
            util.autowarn(dropdown)
            dropdown.on_click(self.on_click(key))
        self.layout = bokeh.layouts.column(
            self.dropdowns["pattern"],
            self.dropdowns["variable"],
            self.dropdowns["initial_time"],
            self.dropdowns["valid_time"],
            self.dropdowns["pressure"])

        super().__init__()

        # Connect state changes to render
        self.subscribe(self.render)

    def render(self, state):
        """Configure dropdown menus"""
        variables = self.database.variables(pattern=state.pattern)
        self.dropdowns["variable"].menu = self.menu(variables)

        initial_times = self.database.initial_times(
            pattern=state.pattern)
        self.dropdowns["initial_time"].menu = self.menu(initial_times)

        if state.initial_time is not None:
            pressures = self.database.pressures(
                pattern=state.pattern,
                variable=state.variable,
                initial_time=state.initial_time)
            pressures = reversed(pressures)
            self.dropdowns["pressure"].menu = self.menu(pressures, self.hpa)

            valid_times = self.database.valid_times(
                pattern=state.pattern,
                variable=state.variable,
                initial_time=state.initial_time)
            valid_times = sorted(set(valid_times))
            self.dropdowns["valid_time"].menu = self.menu(valid_times)

    def on_click(self, key):
        """Wire up bokeh on_click callbacks to State changes"""
        def callback(value):
            state = next_state(self.state, **{key: value})
            self.notify(state)
            self.state = state
        return callback

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
