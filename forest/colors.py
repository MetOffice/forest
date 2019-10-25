"""
Helpers to choose color palette(s), limits etc.
"""
import bokeh.palettes
import bokeh.colors
import bokeh.layouts
from forest.db.util import autolabel
from numpy import linspace


class Controls(object):
    def __init__(self, color_mapper, name, number):
        self.name = name
        self.number = number
        self.palettes = bokeh.palettes.all_palettes
        self.color_mapper = color_mapper
        try:  # FCL: Refresh palette from Controls call
            self.color_mapper.palette = bokeh.palettes.all_palettes[name][number]
        except KeyError:  # If it doesn't have a palette with this number of steps, choose the max number of steps
            # (though alternatively we could also pick the nearest value to the one specified)
            self.color_mapper.palette = bokeh.palettes.all_palettes[name][max(bokeh.palettes.all_palettes[name].keys())]

        self.cbar_min = color_mapper.low
        self.cbar_max = color_mapper.high

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

        # Option to set customised discrete colourscale intervals
        # (None of this is actually linked up to the colour bar configuration... )
        # length of list depending on setting in numbers
        # Want list input only appearing when cbar_discrete is selected --> Done
        # Want cbar_min / max to disappear (those values are specified as first and last values of interval list --> Done
        # Want list input to appear below the discrete checkbox (and could do with a better display option...)

        self.cbar_discrete = False
        # Intervals default to linear spacing of min/max with the number of
        # intervals epending on number set in colour steps setting
        self.cbar_discrete_list = linspace(self.cbar_min,self.cbar_max,num=self.number)
        self.cbar_discrete_checkbox = bokeh.models.CheckboxGroup(
            labels=["Customise Colourbar Intervals"],
            active=[])
        self.cbar_discrete_checkbox.on_change("active", self.on_cbar_discrete)
        self.cbar_discrete_input = bokeh.models.TextInput(
            title="Thresholds for Colourbar Intervals",
            value=str(self.cbar_discrete_list))


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
        self.cbar_min_input = bokeh.models.TextInput(
                title="Colourbar Min:",
                value=str(self.cbar_min))
        self.cbar_max_input = bokeh.models.TextInput(
                title="Colourbar Max:",
                value=str(self.cbar_max))

        # Invisible color settings
        self.invisible_on = False
        # *** Want the default values to pick up data min/max or whatever the current colourbar extention defaults to
        # *** Want the upper and lower boxes to only appear when the threshold masking is actually selected
        self.lower = self.cbar_min
        self.upper = self.cbar_max
        self.invisible_checkbox = bokeh.models.CheckboxGroup(
            labels=["Hide data below/above threshold"],
            active=[])
        self.invisible_checkbox.on_change("active",
                self.on_invisible_checkbox)
        self.invisible_input_lower = bokeh.models.TextInput(
                title="Lower Threshold:",
                value=str(self.cbar_min))
        self.invisible_input_lower.on_change("value",
                self.on_invisible_input_lower)
        self.invisible_input_upper = bokeh.models.TextInput(
                title="Upper Threshold:",
                value=str(self.cbar_max))
        self.invisible_input_upper.on_change("value",
                self.on_invisible_input_upper)


        self.default_widges = [self.names,
                self.numbers,
                self.cbar_discrete_checkbox,
                self.checkbox,
                self.invisible_checkbox]
        self.invisible_show = [self.invisible_input_lower,
                               self.invisible_input_upper]
        self.discrete_show = [self.cbar_discrete_input]
        self.min_max_show = [self.cbar_min_input, self.cbar_max_input]
        self.layout = bokeh.layouts.column(self.default_widges+self.min_max_show)
        # self.layout = bokeh.layouts.column(
        #         self.names,
        #         self.numbers,
        #         self.checkbox,
        #         self.cbar_min_input,
        #         self.cbar_max_input,
        #         self.invisible_checkbox,
        #         self.invisible_input_lower,
        #         self.invisible_input_upper)

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

    def on_cbar_discrete(self, attr, old, new):
        if len(new) == 1:
            self.cbar_discrete = True
            # Insert discrete value input and remove cbar min/max input
            self.layout.children = [lc for lc in self.layout.children + self.discrete_show
                                    if lc not in self.min_max_show]
        else:
            self.cbar_discrete = False
            self.layout.children = [lc for lc in self.layout.children + self.min_max_show
                                    if lc not in self.discrete_show]
        self.render()

    def on_cbar_discrete_input(self, attr, old, new):
        self.cbar_discrete_list = new
        self.render()

    def on_invisible_checkbox(self, attr, old, new):
        # Input fields currently not inserted in the most sensible order
        if len(new) == 1:
            self.invisible_on = True
            self.layout.children = self.layout.children + self.invisible_show
        else:
            self.invisible_on = False
            self.layout.children = [lc for lc in self.layout.children
                                    if lc not in self.invisible_show]
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
