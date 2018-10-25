"""Minimal Forest implementation"""
import os
import yaml
import bokeh.plotting
import bokeh.themes
import numpy as np
import jinja2


script_dir = os.path.dirname(__file__)
env = jinja2.Environment(loader=jinja2.FileSystemLoader(script_dir))


PARAMETERS = [
    "Precipitation",
    "Air temperature",
    "Wind vectors",
    "Wind and MSLP",
    "Wind streamlines",
    "MSLP",
    "Cloud fraction",
    "Accum. precip. 3hr",
    "Accum. precip. 6hr",
    "Accum. precip. 12hr",
    "Accum. precip. 24hr",
]


def load_app(config_file):
    return App.load(config_file)


class App(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @classmethod
    def load(cls, config_file):
        with open(config_file) as stream:
            settings = yaml.load(stream)
        return cls(**settings)

    def __call__(self, document):
        app(document, **self.kwargs)


def app(document, title=None, regions=None, models=None):
    figure = bokeh.plotting.figure(
        sizing_mode="stretch_both",
        name="map",
        title=title)
    source = bokeh.models.ColumnDataSource(dict(
        x=[1, 2, 3],
        y=[1, 2, 3],
        size=[5, 5, 5]
    ))
    figure.circle(x="x", y="y", size="size", source=source)
    def on_click():
        size = np.asarray(source.data["size"])
        if max(size) > 20:
            size = size - 10
        else:
            size = size + 5
        source.data["size"] = size
    models = [item["name"] for item in models]
    regions = [item["name"] for item in regions]
    region_drop_down = drop_down(regions)
    region_drop_down.value = encode(regions[0])
    def on_change(attr, old, new):
        print("change: {} {} {}".format(attr, old, new))
        print(decode(regions, new))
    region_drop_down.on_change("value", on_change)
    parameters = PARAMETERS
    document.add_root(controls(models, region_drop_down, parameters))
    document.add_root(figure)

    filename = os.path.join(script_dir, "theme.yaml")
    document.theme = bokeh.themes.Theme(filename=filename)
    document.title = title
    document.template = env.get_template("templates/index.html")


def controls(models, region_drop_down, parameters):
    checkbox = bokeh.models.CheckboxGroup(labels=["Link plots", "Activate slider"], active=[0])
    left_child = bokeh.layouts.column(
            plus_minus_row("Time"),
            plus_minus_row("Forecast"),
            plus_minus_row("Model run"),
            drop_down(models, models[0]))
    right_child = bokeh.layouts.column(
            plus_minus_row("Time"),
            plus_minus_row("Forecast"),
            plus_minus_row("Model run"),
            drop_down(models, models[1]))
    panels = [
     bokeh.models.Panel(child=left_child, title="Left"),
     bokeh.models.Panel(child=right_child, title="Right")
    ]
    tabs = bokeh.models.Tabs(tabs=panels)
    return bokeh.layouts.column(
            checkbox,
            region_drop_down,
            drop_down(parameters, parameters[0]),
            tabs,
            name="btn")


def plus_minus_row(text):
    return bokeh.layouts.row(p(text), button("+"), button("-"))


def p(text):
    return bokeh.models.Paragraph(text=text, width=80)


def button(label, width=40):
    btn = bokeh.models.Button(
        label=label,
        width=width)
    return btn


def drop_down(names, label=None):
    dd = bokeh.models.Dropdown(menu=[
        (name, encode(name)) for name in names
    ], label=label)
    autolabel(dd)
    return dd


def encode(name):
    return name.lower().replace(" ", "").replace(".", "")


def decode(names, encoded_name):
    for name in names:
        if encode(name) == encoded_name:
            return name


def autolabel(drop_down):
    def on_change(attr, old, new):
        for label, key in drop_down.menu:
            if key == new:
                drop_down.label = label
    drop_down.on_change("value", on_change)
