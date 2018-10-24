"""Minimal Forest implementation"""
import os
import yaml
import bokeh.plotting
import bokeh.themes
import numpy as np
import jinja2


script_dir = os.path.dirname(__file__)
env = jinja2.Environment(loader=jinja2.FileSystemLoader(script_dir))


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
    document.add_root(controls(source, models, regions))
    document.add_root(figure)

    filename = os.path.join(script_dir, "theme.yaml")
    document.theme = bokeh.themes.Theme(filename=filename)
    document.title = title
    document.template = env.get_template("templates/index.html")


def controls(source, models, regions):
    def on_click():
        size = np.asarray(source.data["size"])
        if max(size) > 20:
            size = size - 10
        else:
            size = size + 5
        source.data["size"] = size
    btn1 = bokeh.models.Button(
        label="Circle size")
    btn1.on_click(on_click)
    btn2 = bokeh.models.Button(
        label="Circle size")
    btn2.on_click(on_click)
    column = bokeh.layouts.column(
            btn1, btn2,
            drop_down(models),
            drop_down(regions),
            name="btn")
    return column


def drop_down(items):
    names = [item["name"] for item in items]
    def encode(name):
        return name.lower().replace(" ", "").replace(".", "")
    menu = [
        (name, encode(name)) for name in names
    ]
    return bokeh.models.Dropdown(menu=menu)
