"""Minimal Forest implementation"""
import os
import datetime as dt
import yaml
import bokeh.plotting
import bokeh.themes
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import jinja2
import forest.plot
import forest.geography


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


def cubes(state):
    parameter_state, model_state = state
    parameter = name(parameter_state).lower()
    convention = model_state["format"]
    pattern = model_state["pattern"]
    date = dt.datetime(2018, 10, 21)
    path = file_name(
            date,
            pattern,
            "~/s3/stephen-sea-public-london/model_data/")
    section = forest.stash_section(
            parameter,
            convention)
    item = forest.stash_item(
            parameter,
            convention)
    return forest.load_cube(path, section, item)


def file_name(start_date, pattern, directory):
    directory = os.path.expanduser(directory)
    basename = pattern.format(start_date)
    return os.path.join(directory, basename)


def forest_plot(source):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    def wrapper(cube):
        longitudes = cube.coords('longitude')[0].points
        latitudes = cube.coords('latitude')[0].points
        values = cube.data[0]
        mappable = ax.pcolormesh(
                longitudes,
                latitudes,
                values)
        ni, nj = values.shape
        shape = (ni - 1, nj -1)
        left, right = longitudes.min(), longitudes.max()
        bottom, top = latitudes.min(), latitudes.max()
        x = left
        y = bottom
        dw = right - left
        dh = top - bottom
        image = forest.plot.rgba_from_mappable(
                mappable,
                shape)

        # Smooth high resolution imagery
        max_ni, max_nj = 800, 600
        ni, nj, _ = image.shape
        if (ni > max_ni) or (nj > max_nj):
            image = forest.plot.smooth_image(
                    image, (max_ni, max_nj))

        source.data = {
            "x": [x],
            "y": [y],
            "dw": [dw],
            "dh": [dh],
            "image": [image]
        }
    return wrapper


def app(document, title=None, regions=None, models=None):
    x_range = bokeh.models.Range1d(0, 1, bounds="auto")
    y_range = bokeh.models.Range1d(0, 1, bounds="auto")
    figure = bokeh.plotting.figure(
        sizing_mode="stretch_both",
        name="map",
        title=title,
        x_range=x_range,
        y_range=y_range)
    forest.plot.add_x_axes(figure, "above")
    forest.plot.add_y_axes(figure, "right")

    # Left/right RGBA
    left_rgba = bokeh.models.ColumnDataSource({
        "x": [],
        "y": [],
        "dw": [],
        "dh": [],
        "image": []
    })
    figure.image_rgba(
            x="x",
            y="y",
            dw="dw",
            dh="dh",
            image="image",
            source=left_rgba)
    plot_cube = forest_plot(left_rgba)

    # Coastlines/Borders
    coastlines = bokeh.models.ColumnDataSource({
        "xs": [],
        "ys": []
    })
    figure.multi_line(xs="xs", ys="ys", source=coastlines,
                      color="black")
    borders = bokeh.models.ColumnDataSource({
        "xs": [],
        "ys": []
    })
    figure.multi_line(xs="xs", ys="ys", source=borders,
                      color="grey")

    # Streams
    def any_none(items):
        return any([item is None for item in items])
    left_model_stream = forest.Stream()
    right_model_stream = forest.Stream()
    parameter_stream = forest.Stream()
    stream = forest.rx.CombineLatest(
            parameter_stream,
            left_model_stream)
    cube_stream = stream.filter(any_none).map(cubes)
    cube_stream.map(plot_cube)

    # Reactive figure title
    left_model_stream.map(name).map(edit_title(figure))

    # Models
    names = [name(m) for m in models]

    left_model_drop = drop_down(names)
    on_change = on_change_model(models, left_model_stream)
    left_model_drop.on_change("value", on_change)
    left_model_drop.value = encode(name(models[0]))

    right_model_drop = drop_down(names)
    on_change = on_change_model(models, right_model_stream)
    right_model_drop.on_change("value", on_change)
    right_model_drop.value = encode(name(models[1]))

    # Regions
    region_names = [name(region) for region in regions]
    region_drop = drop_down(region_names)
    on_change = on_change_region(
            x_range,
            y_range,
            coastlines,
            borders,
            regions)
    region_drop.on_change("value", on_change)
    region_drop.value = encode(name(regions[0]))

    # Parameters
    parameters = PARAMETERS
    parameter_drop = drop_down(parameters)
    on_change = on_change_parameter(parameters, parameter_stream)
    parameter_drop.on_change("value", on_change)
    parameter_drop.value = encode(parameters[0])

    document.add_root(controls(
        left_model_drop,
        right_model_drop,
        region_drop,
        parameter_drop))
    document.add_root(figure)

    filename = os.path.join(script_dir, "theme.yaml")
    document.theme = bokeh.themes.Theme(filename=filename)
    document.title = title
    document.template = env.get_template("templates/index.html")


def name(item):
    return item["name"]


def edit_title(figure):
    def wrapper(text):
        figure.title.text = text
    return wrapper


def on_change_model(models, stream):
    names = [name(model) for model in models]
    formats = {name(model): model["file"]["format"]
               for model in models}
    patterns = {name(model): model["file"]["pattern"]
               for model in models}
    def on_change(attr, old, new):
        name = decode(names, new)
        stream.emit({
            "name": name,
            "format": formats[name],
            "pattern": patterns[name],
        })
    return on_change


def on_change_parameter(names, stream):
    def on_change(attr, old, new):
        name = decode(names, new)
        stream.emit({
            "name": name
        })
    return on_change


def on_change_region(
        x_range,
        y_range,
        coastlines,
        borders,
        regions):
    names = [name(region) for region in regions]
    x_ranges = {name(region): lon_range(region)
            for region in regions}
    y_ranges = {name(region): lat_range(region)
            for region in regions}
    def on_change(attr, old, new):
        name = decode(names, new)
        x_start, x_end = x_ranges[name]
        y_start, y_end = y_ranges[name]
        print(name, x_start, x_end, y_start, y_end)
        extent = x_start, x_end, y_start, y_end
        xs, ys = forest.geography.coastlines(extent)
        coastlines.data = {
            "xs": xs,
            "ys": ys
        }
        xs, ys = forest.geography.borders(extent)
        borders.data = {
            "xs": xs,
            "ys": ys
        }
        x_range.start = x_start
        x_range.end = x_end
        y_range.start = y_start
        y_range.end = y_end
    return on_change


def lon_range(region):
    start, end = region["longitude_range"]
    return float(start), float(end)


def lat_range(region):
    start, end = region["latitude_range"]
    return float(start), float(end)


def controls(
        left_model_drop,
        right_model_drop,
        *extra_drops):
    checkbox = bokeh.models.CheckboxGroup(labels=["Link plots", "Activate slider"], active=[0])
    left_child = bokeh.layouts.column(
            plus_minus_row("Time"),
            plus_minus_row("Forecast"),
            plus_minus_row("Model run"),
            left_model_drop)
    right_child = bokeh.layouts.column(
            plus_minus_row("Time"),
            plus_minus_row("Forecast"),
            plus_minus_row("Model run"),
            right_model_drop)
    panels = [
     bokeh.models.Panel(child=left_child, title="Left"),
     bokeh.models.Panel(child=right_child, title="Right")
    ]
    tabs = bokeh.models.Tabs(tabs=panels)
    return bokeh.layouts.column(
            checkbox,
            *extra_drops,
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
