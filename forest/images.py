import bokeh.models
import bokeh.layouts
import numpy as np
from util import select, Observable


class Controls(Observable):
    def __init__(self, menu):
        self.menu = menu
        self.models = {}
        self.flags = {}
        self.default_flags = [False, False, False]

        self.state = {}
        self.previous_state = None
        self.renderers = []
        self._labels = ["Show"]
        self.groups = []

        add = bokeh.models.Button(label="Add", width=50)
        remove = bokeh.models.Button(label="Remove", width=50)
        self.column = bokeh.layouts.column(
            bokeh.layouts.row(add, remove)
        )
        self.add_row()
        add.on_click(self.add_row)
        remove.on_click(self.remove_row)
        super().__init__()

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels
        for g in self.groups:
            g.labels = labels

    def add_row(self):
        i = self.rows
        dropdown = bokeh.models.Dropdown(
                menu=self.menu,
                label="Model/observation",
                width=150)
        dropdown.on_click(select(dropdown))
        dropdown.on_change('value', self.on_dropdown(i))
        group = bokeh.models.CheckboxButtonGroup(
                labels=self.labels)
        group.on_change("active", self.on_radio(i))
        self.groups.append(group)
        row = bokeh.layouts.row(dropdown, group)
        self.column.children.insert(-1, row)

    def remove_row(self):
        if len(self.column.children) > 2:
            i = self.rows - 1
            self.models.pop(i, None)
            self.flags.pop(i, None)
            self.column.children.pop(-2)
            self.render()

    @property
    def rows(self):
        return len(self.column.children) - 1

    def on_dropdown(self, i):
        def wrapper(attr, old, new):
            if old != new:
                self.models[i] = new
                self.render()
        return wrapper

    def on_radio(self, i):
        def wrapper(attr, old, new):
            if i not in self.flags:
                self.flags[i] = list(self.default_flags)

            flags = self.flags[i]
            for j in old:
                if j not in new:
                    flags[j] = False
            for j in new:
                if j not in old:
                    flags[j] = True
            self.render()
        return wrapper

    def render(self):
        self.announce(self.combine(self.models, self.flags))

    @staticmethod
    def combine(models, flags):
        agg = {}
        for k in set(models.keys()).intersection(
                set(flags.keys())):
            if models[k] in agg:
                agg[models[k]].append(flags[k])
            else:
                agg[models[k]] = [flags[k]]
        combined = {}
        for k, v in agg.items():
            if len(agg[k]) > 1:
                combined[k] = np.logical_or(*agg[k]).tolist()
            else:
                combined[k] = agg[k][0]
        return combined
