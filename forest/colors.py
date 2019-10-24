"""
Helpers to choose color palette(s), limits etc.
"""
import bokeh.palettes
import bokeh.colors
import bokeh.layouts
from forest.db.util import autolabel


class Controls(object):
    def __init__(self, color_mapper, name, number, cbar_min, cbar_max):
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
                label="Colour Steps",#Was called N
                value=str(self.number),
                menu=self.numbers_menu(numbers))
        autolabel(self.numbers)
        self.numbers.on_change("value", self.on_number)

        self.reverse = False
        #Tickbox more intuitive for On/Off selection
        # - easier to see what is selected
        # - easier to see difference between on/off switch and dropdown menu
        self.checkbox = bokeh.models.CheckboxGroup(
            labels=["Reverse Colourbar"],
            active=[])
        self.checkbox.on_change("active", self.on_reverse)

        # Color bar min/max settings
        # *** Want the default values to pick up data min/max or whatever it currently defaults to
        self.cbar_min = cbar_min
        self.cbar_max = cbar_max
        self.cbar_min_input = bokeh.models.TextInput(
                title="Colourbar Min:",
                value=str(cbar_min))
        self.cbar_max_input = bokeh.models.TextInput(
                title="Colourbar Max:",
                value=str(cbar_max))

        # Invisible color settings
        self.invisible_on = False
        # *** Want the default values to pick up data min/max or whatever the current colourbar extention defaults to
        # *** Want the upper and lower boxes to only appear when the threshold masking is actually selected
        self.lower = cbar_min
        self.upper = cbar_max
        self.invisible_checkbox = bokeh.models.CheckboxGroup(
            labels=["Hide data below/above threshold"],
            active=[])
        self.invisible_checkbox.on_change("active",
                self.on_invisible_checkbox)
        self.invisible_input_lower = bokeh.models.TextInput(
                title="Lower Threshold:",
                value=str(cbar_min))
        self.invisible_input_lower.on_change("value",
                self.on_invisible_input_lower)
        self.invisible_input_upper = bokeh.models.TextInput(
                title="Upper Threshold:",
                value=str(cbar_max))
        self.invisible_input_upper.on_change("value",
                self.on_invisible_input_upper)

        self.layout = bokeh.layouts.column(
                self.names,
                self.numbers,
                self.checkbox,
                self.cbar_min_input,
                self.cbar_max_input,
                self.invisible_checkbox,
                self.invisible_input_lower,
                self.invisible_input_upper)

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

    def on_cbar_min_input(self, attr, old, new):
        self.cbar_min = float(new)
        self.render()

    def on_cbar_max_input(self, attr, old, new):
        self.cbar_max = float(new)
        self.render()

    def on_invisible_checkbox(self, attr, old, new):
        if len(new) == 1:
            self.invisible_on = True
        else:
            self.invisible_on = False
        self.render()

    def on_invisible_input_lower(self, attr, old, new):
        self.lower = float(new)
        self.render()

    def on_invisible_input_upper(self, attr, old, new):
        self.upper = float(new)
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
            lower = self.lower
            color = bokeh.colors.RGB(0, 0, 0, a=0)
            self.color_mapper.low_color = color
            self.color_mapper.low = lower
        self.color_mapper.palette = palette
