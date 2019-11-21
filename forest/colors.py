"""
Helpers to choose color palette(s), limits etc.
"""
import bokeh.palettes
import bokeh.colors
import bokeh.layouts
import numpy as np
from forest.observe import Observable
from forest.redux import middleware
from forest.db.util import autolabel


SET_PALETTE = "SET_PALETTE"


def set_fixed(flag):
    return {"kind": SET_PALETTE, "payload": {"key": "fixed", "value": flag}}


def set_reverse(flag):
    return {"kind": SET_PALETTE, "payload": {"key": "reverse", "value": flag}}


def set_palette_name(name):
    return {"kind": SET_PALETTE, "payload": {"key": "name", "value": name}}


def set_palette_number(number):
    return {"kind": SET_PALETTE, "payload": {"key": "number", "value": number}}


def set_palette_numbers(numbers):
    return {"kind": SET_PALETTE, "payload": {"key": "numbers", "value": numbers}}


def set_palette_names(names):
    return {"kind": SET_PALETTE, "payload": {"key": "names", "value": names}}


def reducer(state, action):
    kind = action["kind"]
    if kind == SET_PALETTE:
        key, value =action["payload"]["key"], action["payload"]["value"]
        settings = state.get("colorbar", {})
        settings[key] = value
        state["colorbar"] = settings
    return state


@middleware
def palettes(store, next_dispatch, action):
    kind = action["kind"]
    if kind == SET_PALETTE:
        key = action["payload"]["key"]
        value = action["payload"]["value"]
        if key == "name":
            numbers = palette_numbers(value)
            next_dispatch(set_palette_numbers(numbers))
            if "colorbar" in store.state:
                if "number" in store.state["colorbar"]:
                    number = store.state["colorbar"]["number"]
                    if number not in numbers:
                        next_dispatch(set_palette_number(max(numbers)))
        next_dispatch(action)
    else:
        next_dispatch(action)


def palette_numbers(name):
    return list(sorted(bokeh.palettes.all_palettes[name].keys()))


class MapperLimits(Observable):
    def __init__(self, sources, color_mapper, fixed=False):
        self.fixed = fixed
        self.sources = sources
        for source in self.sources:
            source.on_change("data", self.on_source_change)
        self.color_mapper = color_mapper
        self.low_input = bokeh.models.TextInput(title="Low:")
        self.low_input.on_change("value",
                self.change(color_mapper, "low", float))
        self.color_mapper.on_change("low",
                self.change(self.low_input, "value", str))
        self.high_input = bokeh.models.TextInput(title="High:")
        self.high_input.on_change("value",
                self.change(color_mapper, "high", float))
        self.color_mapper.on_change("high",
                self.change(self.high_input, "value", str))
        self.checkbox = bokeh.models.CheckboxGroup(
                labels=["Fixed"],
                active=[])
        self.checkbox.on_change("active", self.on_checkbox_change)
        super().__init__()

    def on_checkbox_change(self, attr, old, new):
        if len(new) == 1:
            self.fixed = True
            self.notify(set_fixed(True))
        else:
            self.fixed = False
            self.notify(set_fixed(False))

    def on_source_change(self, attr, old, new):
        if self.fixed:
            return
        images = []
        for source in self.sources:
            if len(source.data["image"]) == 0:
                continue
            images.append(source.data["image"][0])
        if len(images) > 0:
            low = np.min([np.min(x) for x in images])
            high = np.max([np.max(x) for x in images])
            self.color_mapper.low = low
            self.color_mapper.high = high
            self.color_mapper.low_color = bokeh.colors.RGB(0, 0, 0, a=0)
            self.color_mapper.high_color = bokeh.colors.RGB(0, 0, 0, a=0)

    @staticmethod
    def change(widget, prop, dtype):
        def wrapper(attr, old, new):
            if old == new:
                return
            if getattr(widget, prop) == dtype(new):
                return
            setattr(widget, prop, dtype(new))
        return wrapper


class Invisible:
    """Control transparency thresholds"""
    def __init__(self, color_mapper):
        self.color_mapper = color_mapper
        self.invisible_on = False
        self.low = 0
        self.invisible_checkbox = bokeh.models.CheckboxButtonGroup(
            labels=["Invisible"],
            active=[])
        self.invisible_checkbox.on_change("active",
                self.on_invisible_checkbox)
        self.invisible_input = bokeh.models.TextInput(
                title="Low:",
                value="0")
        self.invisible_input.on_change("value",
                self.on_invisible_input)
        self.layout = bokeh.layouts.column(
                self.invisible_checkbox,
                self.invisible_input)

    def on_invisible_checkbox(self, attr, old, new):
        if len(new) == 1:
            self.invisible_on = True
        else:
            self.invisible_on = False

    def on_invisible_input(self, attr, old, new):
        self.low = float(new)

    def render(self):
        if self.invisible_on:
            low = self.low
            color = bokeh.colors.RGB(0, 0, 0, a=0)
            self.color_mapper.low_color = color
            self.color_mapper.low = low


class Controls(Observable):
    def __init__(self, color_mapper):
        self.color_mapper = color_mapper
        self.dropdowns = {
            "names": bokeh.models.Dropdown(label="Palettes"),
            "numbers": bokeh.models.Dropdown(label="N")
        }
        self.dropdowns["names"].on_change("value", self.on_name)
        self.dropdowns["numbers"].on_change("value", self.on_number)

        self.checkbox = bokeh.models.CheckboxButtonGroup(
            labels=["Reverse"],
            active=[])
        self.checkbox.on_change("active", self.on_reverse)

        self.layout = bokeh.layouts.column(
                self.dropdowns["names"],
                self.dropdowns["numbers"],
                self.checkbox)
        super().__init__()

    def on_name(self, attr, old, new):
        self.notify(set_palette_name(new))

    def on_number(self, attr, old, new):
        self.notify(set_palette_number(int(new)))

    def on_reverse(self, attr, old, new):
        self.notify(set_reverse(len(new) == 1))

    def render(self, state):
        if "colorbar" not in state:
            return

        settings = state["colorbar"]
        if "name" in settings:
            self.dropdowns["names"].value = settings["name"]
        if "number" in settings:
            self.dropdowns["numbers"].value = str(settings["number"])
        if ("name" in settings) and ("number" in settings):
            name = settings["name"]
            number = settings["number"]
            reverse = settings.get("reverse", False)
            palette = self.palette(name, number)
            if reverse:
                palette = palette[::-1]
            self.color_mapper.palette = palette
        if "names" in settings:
            values = settings["names"]
            self.dropdowns["names"].menu = list(zip(values, values))
        if "numbers" in settings:
            values = [str(n) for n in settings["numbers"]]
            self.dropdowns["numbers"].menu = list(zip(values, values))

    @staticmethod
    def palette(name, number):
        return bokeh.palettes.all_palettes[name][number]
