"""
Helpers to choose color palette(s), limits etc.
"""
import bokeh.palettes
import bokeh.colors
import bokeh.layouts
from forest.db.util import autolabel


class Controls(object):
    def __init__(self, color_mapper, name, number):
        self.name = name
        self.number = number
        self.palettes = bokeh.palettes.all_palettes
        self.color_mapper = color_mapper

        names = sorted(self.palettes.keys())
        menu = list(zip(names, names))
        self.names = bokeh.models.Dropdown(
                label="Palettes",
                value=self.name,
                menu=menu)
        autolabel(self.names)
        self.names.on_change("value", self.on_name)

        numbers = sorted(self.palettes[self.name].keys())
        self.numbers = bokeh.models.Dropdown(
                label="N",
                value=str(self.number),
                menu=self.numbers_menu(numbers))
        autolabel(self.numbers)
        self.numbers.on_change("value", self.on_number)

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
                self.names,
                self.numbers,
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
        self.numbers.menu = self.numbers_menu(numbers)
        self.numbers.value = str(self.number)
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
