"""User-defined border overlays"""
import bokeh.models
import bokeh.layouts
import forest.actions
from forest import data
from forest.observe import Observable
from forest.state import State
from forest.mark import component


class View:
    def __init__(self):
        self.sources = {
            "borders": bokeh.models.ColumnDataSource(data.BORDERS),
            "coastlines": bokeh.models.ColumnDataSource(data.COASTLINES),
            "disputed": bokeh.models.ColumnDataSource(data.DISPUTED),
            "lakes": bokeh.models.ColumnDataSource(data.LAKES),
        }
        self.renderers = {
            "all": [],
            "coastline": []
        }

    def add_figure(self, figure):
        # Lakes
        renderer = figure.multi_line(xs="xs",
                                     ys="ys",
                                     source=self.sources["lakes"],
                                     color="lightblue")
        self.renderers["all"].append(renderer)

        # Coastline
        renderer = figure.multi_line(xs="xs",
                                     ys="ys",
                                     source=self.sources["coastlines"],
                                     color="black")
        self.renderers["coastline"].append(renderer)
        self.renderers["all"].append(renderer)

        # Borders
        renderer = figure.multi_line(xs="xs",
                                     ys="ys",
                                     source=self.sources["borders"],
                                     color="black")
        self.renderers["coastline"].append(renderer)
        self.renderers["all"].append(renderer)

        # Disputed borders
        renderer = figure.multi_line(xs="xs",
                                     ys="ys",
                                     source=self.sources["disputed"],
                                     color="red")
        self.renderers["all"].append(renderer)

    def connect(self, store):
        store.add_subscriber(self.render)

    def render(self, state):
        if isinstance(state, dict):
            state = State.from_dict(state)
        for renderer in self.renderers["all"]:
            renderer.visible = state.borders.visible
        for renderer in self.renderers["coastline"]:
            renderer.glyph.line_color = state.borders.line_color


@component
class UI(Observable):
    def __init__(self):
        self.checkbox = bokeh.models.CheckboxGroup(
                labels=["Coastlines"],
                active=[0],
                width=135)
        self.checkbox.on_change("active", self.on_checkbox)
        self._please_specify = "Select color"
        self._colors = ["black", "white"]
        self._options = [self._please_specify] + [color.capitalize()
                                                  for color in self._colors]
        self.select = bokeh.models.Select(
                options=self._options,
                width=100)
        self.select.on_change("value", self.on_select)
        self.layout = bokeh.layouts.row(self.checkbox,
                                        self.select)
        super().__init__()

    def connect(self, store):
        self.add_subscriber(store.dispatch)
        store.add_subscriber(self.render)

    def render(self, state):
        if isinstance(state, dict):
            state = State.from_dict(state)
        if state.borders.visible:
            self.checkbox.active = [0]
        else:
            self.checkbox.active = []

        # Find/set value
        for option in self._options:
            if state.borders.line_color.lower() == option.lower():
                self.select.value = option
                break

    def on_checkbox(self, attr, old, new):
        action = forest.actions.set_borders_visible(len(new) == 1)
        self.notify(action.to_dict())

    def on_select(self, attr, old, new):
        if new.lower() != self._please_specify.lower():
            action = forest.actions.set_borders_line_color(new.lower())
            self.notify(action.to_dict())
