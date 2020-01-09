import copy
import bokeh.models
import bokeh.layouts
import numpy as np
from forest.observe import Observable
from forest.db.util import autolabel


SET_FIGURES = 3


def set_figures(n):
    return {"kind": SET_FIGURES, "payload": n}


def reducer(state, action):
    state = copy.deepcopy(state)
    kind = action["kind"]
    if kind == SET_FIGURES:
        state["figures"] = action["payload"]
    return state


class FigureRow:
    def __init__(self, figures):
        self.figures = figures
        self.layout = bokeh.layouts.row(*figures,
                sizing_mode="stretch_both")
        self.layout.children = [self.figures[0]]  # Trick to keep correct sizing modes

    def render(self, n):
        if int(n) == 1:
            self.layout.children = [
                    self.figures[0]]
        elif int(n) == 2:
            self.layout.children = [
                    self.figures[0],
                    self.figures[1]]
        elif int(n) == 3:
            self.layout.children = [
                    self.figures[0],
                    self.figures[1],
                    self.figures[2]]


class LeftCenterRight:
    def __init__(self, controls):
        self.controls = controls

    def render(self, n):
        if int(n) == 1:
            self.controls.labels = ["Show"]
        elif int(n) == 2:
            self.controls.labels = ["L", "R"]
        elif int(n) == 3:
            self.controls.labels = ["L", "C", "R"]


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
        self.dropdowns = []

        add = bokeh.models.Button(label="Add", width=50)
        remove = bokeh.models.Button(label="Remove", width=50)
        self.column = bokeh.layouts.column(
            bokeh.layouts.row(add, remove)
        )
        self.add_row()
        add.on_click(self.add_row)
        remove.on_click(self.remove_row)
        super().__init__()

    def select(self, name):
        """Select particular layers and visibility states"""
        self.groups[0].active = [0]
        dropdown = self.dropdowns[0]
        for k, v in dropdown.menu:
            if k == name:
                dropdown.value = v

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
                width=230,)
        autolabel(dropdown)
        dropdown.on_change('value', self.on_dropdown(i))
        self.dropdowns.append(dropdown)
        group = bokeh.models.CheckboxButtonGroup(
                labels=self.labels,
                width=50)
        group.on_change("active", self.on_radio(i))
        self.groups.append(group)
        row = bokeh.layouts.row(dropdown, group)
        self.column.children.insert(-1, row)

    def remove_row(self):
        if len(self.column.children) > 2:
            i = self.rows - 1
            self.models.pop(i, None)
            self.flags.pop(i, None)
            self.dropdowns.pop(-1)
            self.groups.pop(-1)
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
        self.notify(self.combine(self.models, self.flags))

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


class Artist(object):
    def __init__(self, viewers, renderers):
        self.viewers = viewers
        self.renderers = renderers
        self.visible_state = None
        self.state = None

    def on_visible(self, visible_state):
        if self.visible_state is not None:
            # Hide deselected states
            lost_items = (
                    set(self.flatten(self.visible_state)) -
                    set(self.flatten(visible_state)))
            for key, i, _ in lost_items:
                self.renderers[key][i].visible = False

        # Sync visible states with menu choices
        states = set(self.flatten(visible_state))
        hidden = [(i, j) for i, j, v in states if not v]
        visible = [(i, j) for i, j, v in states if v]
        for i, j in hidden:
            self.renderers[i][j].visible = False
        for i, j in visible:
            self.renderers[i][j].visible = True

        self.visible_state = dict(visible_state)
        self.render()

    @staticmethod
    def flatten(state):
        items = []
        for key, flags in state.items():
            items += [(key, i, f) for i, f in enumerate(flags)]
        return items

    def on_state(self, state):
        # print("Artist: {}".format(state))
        self.state = state
        self.render()

    def render(self):
        if self.visible_state is None:
            return
        if self.state is None:
            return
        for name in self.visible_state:
            viewer = self.viewers[name]
            viewer.render(self.state)
