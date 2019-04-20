"""
Helpers to choose color palette(s), limits etc.
"""
import bokeh.palettes
import bokeh.layouts
from util import select


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
        self.names.on_click(select(self.names))
        self.names.on_click(self.on_name)

        numbers = sorted(self.palettes[self.name].keys())
        self.numbers = bokeh.models.Dropdown(
                label="N",
                value=str(self.number),
                menu=self.numbers_menu(numbers))
        self.numbers.on_click(select(self.numbers))
        self.numbers.on_click(self.on_number)

        self.reverse = False
        self.checkbox = bokeh.models.CheckboxButtonGroup(
            labels=["Reverse"],
            active=[])
        self.checkbox.on_change("active", self.on_reverse)

        self.layout = bokeh.layouts.column(
                self.names,
                self.numbers,
                self.checkbox)

    def on_name(self, name):
        self.name = name
        numbers = sorted(self.palettes[name].keys())
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

    def on_number(self, number):
        self.number = int(number)
        self.render()

    def on_reverse(self, attr, old, new):
        print(attr, old, new)
        if len(new) == 1:
            self.reverse = True
        else:
            self.reverse = False
        self.render()

    def render(self):
        if self.name is None:
            return
        if self.number is None:
            return
        palette = self.palettes[self.name][self.number]
        if self.reverse:
            palette = list(reversed(palette))
        print(palette)
        self.color_mapper.palette = palette
