"""Widgets responsible for comparisons"""
from collections import defaultdict
import bokeh.models
from util import select


def table(*args, labels=None):
    return Table(RowFactory(*args, labels=labels))


class Table(object):
    def __init__(self, row_factory):
        self.rows = []
        self.row_factory = row_factory
        add_btn = bokeh.models.Button(
                label="Add",
                width=50)
        add_btn.on_click(self.add_row)
        remove_btn = bokeh.models.Button(
                label="Remove",
                width=50)
        remove_btn.on_click(self.remove_row)
        self.layout = bokeh.layouts.column(
            bokeh.layouts.row(add_btn, remove_btn),
            width=300)

    def add_row(self):
        row = self.row_factory()
        self.layout.children.insert(-1, row.layout)
        self.rows.append(row)

    def remove_row(self):
        if len(self.layout.children) > 2:
            self.rows[-1].remove()
            self.layout.children.pop(-2)


class RowFactory(object):
    def __init__(self, *args, labels=None):
        self.args = args
        self.labels = labels

    def __call__(self):
        return Row(*self.args, labels=self.labels)


class Row(object):
    def __init__(self, names, views, figures, labels):
        self.rms = defaultdict(list)
        self.figures = figures
        menu = [(n, n) for n in names]
        dropdown = bokeh.models.Dropdown(
                menu=menu,
                label="Model/observation",
                width=110)
        dropdown.on_click(select(dropdown))
        dropdown.on_click(self.on_view)
        self.views = dict(zip(names, views))
        self.view = None
        self.group = bokeh.models.CheckboxButtonGroup(
                labels=labels)
        self.group.on_change("active", self.on_group)
        self.layout = bokeh.layouts.row(
                bokeh.layouts.widgetbox(
                    dropdown,
                    width=130),
                self.group)

    @property
    def labels(self):
        return self.group.labels

    @labels.setter
    def labels(self, values):
        self.group.labels = values

    def on_view(self, name):
        for i in self.group.active:
            self.remove(i)
        self.view = self.views[name]
        for i in self.group.active:
            self.add(i)

    def on_group(self, attr, old, new):
        for i in set(old) - set(new):
            self.remove(i)
        for i in set(new) - set(old):
            self.add(i)

    def add(self, i):
        if self.view is None:
            return
        figure = self.figures[i]
        self.rms[i].append(
            self.view.add_figure(figure))

    def remove(self, i):
        for rm in self.rms[i]:
            rm()
        self.rms[i] = []
