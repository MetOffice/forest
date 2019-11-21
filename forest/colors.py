"""
Helpers to choose color palette(s), limits etc.
"""
import bokeh.palettes
import bokeh.colors
import bokeh.layouts
import numpy as np
from forest.observe import Observable
from forest.db.util import autolabel


SET_FIXED = "SET_FIXED"


def fixed_on():
    return {"kind": SET_FIXED, "payload": {"status": "on"}}


def fixed_off():
    return {"kind": SET_FIXED, "payload": {"status": "off"}}


def reducer(state, action):
    if action["kind"] == SET_FIXED:
        state["colorbar"] = {"fixed": action["payload"]["status"] == "on"}
    return state


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
            self.notify(fixed_on())
        else:
            self.fixed = False
            self.notify(fixed_off())

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


class Controls(object):
    def __init__(self, color_mapper, name, number):
        self.name = name
        self.number = number
        self.palettes = bokeh.palettes.all_palettes
        self.color_mapper = color_mapper

        names = sorted(self.palettes.keys())
        menu = list(zip(names, names))

        self.dropdowns = {}
        self.dropdowns["names"] = bokeh.models.Dropdown(
                label="Palettes",
                value=self.name,
                menu=menu)
        autolabel(self.dropdowns["names"])
        self.dropdowns["names"].on_change("value", self.on_name)

        numbers = sorted(self.palettes[self.name].keys())
        self.dropdowns["numbers"] = bokeh.models.Dropdown(
                label="N",
                value=str(self.number),
                menu=self.numbers_menu(numbers))
        autolabel(self.dropdowns["numbers"])
        self.dropdowns["numbers"].on_change("value", self.on_number)

        self.reverse = False
        self.checkbox = bokeh.models.CheckboxButtonGroup(
            labels=["Reverse"],
            active=[])
        self.checkbox.on_change("active", self.on_reverse)

        # Invisible color settings
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
                self.dropdowns["names"],
                self.dropdowns["numbers"],
                self.checkbox,
                self.invisible_checkbox,
                self.invisible_input)

    def on_name(self, attr, old, new):
        self.name = new
        numbers = sorted(self.palettes[self.name].keys())
        if self.number is None:
            self.number = numbers[-1]
        elif self.number not in numbers:
            self.number = numbers[-1]
        self.dropdowns["numbers"].menu = self.numbers_menu(numbers)
        self.dropdowns["numbers"].value = str(self.number)
        self.render()

    def numbers_menu(self, numbers):
        labels = [str(n) for n in numbers]
        return list(zip(labels, labels))

    def on_number(self, attr, old, new):
        self.number = int(new)
        self.render()

    def on_reverse(self, attr, old, new):
        if len(new) == 1:
            self.reverse = True
        else:
            self.reverse = False
        self.render()

    def on_invisible_checkbox(self, attr, old, new):
        if len(new) == 1:
            self.invisible_on = True
        else:
            self.invisible_on = False
        self.render()

    def on_invisible_input(self, attr, old, new):
        self.low = float(new)
        self.render()

    def render(self):
        if self.name is None:
            return
        if self.number is None:
            return
        palette = self.palettes[self.name][self.number]
        if self.reverse:
            palette = list(reversed(palette))
        if self.invisible_on:
            low = self.low
            color = bokeh.colors.RGB(0, 0, 0, a=0)
            self.color_mapper.low_color = color
            self.color_mapper.low = low
        self.color_mapper.palette = palette
