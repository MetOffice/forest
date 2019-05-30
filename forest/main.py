import wind  # meta-programming to inject figure.barbs()
import bokeh.plotting
import bokeh.events
import numpy as np
import os
import satellite
import data
import view
import images
import earth_networks
import rdt
import geo
import picker
import colors
import compare
import db
import parse_args
from util import Observable, select, initial_time
from collections import defaultdict, namedtuple
import datetime as dt


def main():
    args = parse_args.parse_args()
    with open(args.config_file) as stream:
        config = parse_args.load_config(stream)

    # Access latest files
    data.FILE_DB.sync()

    # Full screen map
    lon_range = (0, 30)
    lat_range = (0, 30)
    x_range, y_range = geo.web_mercator(
        lon_range,
        lat_range)
    figure = bokeh.plotting.figure(
        x_range=x_range,
        y_range=y_range,
        x_axis_type="mercator",
        y_axis_type="mercator",
        active_scroll="wheel_zoom")
    tile = bokeh.models.WMTSTileSource(
        url="https://maps.wikimedia.org/osm-intl/{Z}/{X}/{Y}.png",
        attribution=""
    )

    figures = [figure]
    for _ in range(2):
        f = bokeh.plotting.figure(
            x_range=figure.x_range,
            y_range=figure.y_range,
            x_axis_type="mercator",
            y_axis_type="mercator",
            active_scroll="wheel_zoom")
        figures.append(f)

    for f in figures:
        f.axis.visible = False
        f.toolbar.logo = None
        f.toolbar_location = None
        f.min_border = 0
        f.add_tile(tile)

    figure_row = bokeh.layouts.row(*figures,
            sizing_mode="stretch_both")
    figure_row.children = [figures[0]]  # Trick to keep correct sizing modes

    figure_drop = bokeh.models.Dropdown(
            label="Figure",
            menu=[(str(i), str(i)) for i in [1, 2, 3]])

    def on_click(value):
        if int(value) == 1:
            figure_row.children = [
                    figures[0]]
        elif int(value) == 2:
            figure_row.children = [
                    figures[0],
                    figures[1]]
        elif int(value) == 3:
            figure_row.children = [
                    figures[0],
                    figures[1],
                    figures[2]]

    figure_drop.on_click(on_click)

    color_mapper = bokeh.models.LinearColorMapper(
            low=0,
            high=1,
            palette=bokeh.palettes.Plasma[256])
    for figure in figures:
        colorbar = bokeh.models.ColorBar(
            color_mapper=color_mapper,
            orientation="horizontal",
            background_fill_alpha=0.,
            location="bottom_center",
            major_tick_line_color="black",
            bar_line_color="black")
        figure.add_layout(colorbar, 'center')

    artist = Artist(figures, color_mapper)
    renderers = []
    for _, r in artist.renderers.items():
        renderers += r

    image_sources = []
    for name, viewer in artist.viewers.items():
        if isinstance(viewer, (view.UMView, view.GPMView, view.EIDA50)):
            image_sources.append(viewer.source)

    image_loaders = []
    for name, loader in data.LOADERS.items():
        if isinstance(loader, (data.UMLoader, data.GPM)):
            image_loaders.append(loader)

    # Lakes
    for figure in figures:
        add_feature(figure, data.LAKES, color="lightblue")

    features = []
    for figure in figures:
        features += [
            add_feature(figure, data.COASTLINES),
            add_feature(figure, data.BORDERS)]

    # Disputed borders
    for figure in figures:
        add_feature(figure, data.DISPUTED, color="red")

    toggle = bokeh.models.CheckboxButtonGroup(
            labels=["Coastlines"],
            active=[0],
            width=135)

    def on_change(attr, old, new):
        if len(new) == 1:
            for feature in features:
                feature.visible = True
        else:
            for feature in features:
                feature.visible = False

    toggle.on_change("active", on_change)

    dropdown = bokeh.models.Dropdown(
            label="Color",
            menu=[
                ("Black", "black"),
                ("White", "white")],
            width=50)
    dropdown.on_click(select(dropdown))

    def on_click(value):
        for feature in features:
            feature.glyph.line_color = value

    dropdown.on_click(on_click)

    slider = bokeh.models.Slider(
        start=0,
        end=1,
        step=0.1,
        value=1.0,
        show_value=False)
    custom_js = bokeh.models.CustomJS(
            args=dict(renderers=renderers),
            code="""
            renderers.forEach(function (r) {
                r.glyph.global_alpha = cb_obj.value
            })
            """)
    slider.js_on_change("value", custom_js)

    colors_controls = colors.Controls(
            color_mapper, "Plasma", 256)

    mapper_limits = MapperLimits(image_sources, color_mapper)

    menu = [(n, n) for n in data.FILE_DB.names]
    image_controls = images.Controls(menu)

    def on_click(value):
        if int(value) == 1:
            image_controls.labels = ["Show"]
        elif int(value) == 2:
            image_controls.labels = ["L", "R"]
        elif int(value) == 3:
            image_controls.labels = ["L", "C", "R"]

    figure_drop.on_click(on_click)

    field_controls = FieldControls()

    image_controls.subscribe(artist.on_visible)
    field_controls.subscribe(artist.on_field)

    div = bokeh.models.Div(text="", width=10)
    border_row = bokeh.layouts.row(
        bokeh.layouts.column(toggle),
        bokeh.layouts.column(div),
        bokeh.layouts.column(dropdown))

    # Add prototype database controls
    database = db.Database.connect(args.database)
    controls = db.Controls(database, patterns=config.patterns)

    tabs = bokeh.models.Tabs(tabs=[
        bokeh.models.Panel(
            child=controls.layout,
            title="DB"
        ),
        bokeh.models.Panel(
            child=bokeh.layouts.column(
                bokeh.layouts.row(figure_drop),
                image_controls.column),
            title="Compare"),
        bokeh.models.Panel(
            child=bokeh.layouts.column(
                border_row,
                bokeh.layouts.row(slider),
                colors_controls.layout,
                bokeh.layouts.row(mapper_limits.low_input),
                bokeh.layouts.row(mapper_limits.high_input),
                bokeh.layouts.row(mapper_limits.checkbox),
                ),
            title="Settings")
        ])

    # Series sub-figure widget
    series_figure = bokeh.plotting.figure(
                plot_width=400,
                plot_height=200,
                x_axis_type="datetime",
                toolbar_location=None,
                border_fill_alpha=0)
    series_figure.toolbar.logo = None
    series_row = bokeh.layouts.row(
            series_figure,
            name="series")

    def place_marker(figure, source):
        figure.circle(
                x="x",
                y="y",
                color="red",
                source=source)
        def cb(event):
            source.data = {
                    "x": [event.x],
                    "y": [event.y]}
        return cb

    marker_source = bokeh.models.ColumnDataSource({
            "x": [],
            "y": []})

    series = Series(series_figure)
    field_controls.subscribe(series.on_field)
    for f in figures:
        f.on_event(bokeh.events.Tap, series.on_tap)
        f.on_event(bokeh.events.Tap, place_marker(f, marker_source))


    # Minimise controls to ease navigation
    compact_button = bokeh.models.Button(
            label="Compact")
    compact_minus = bokeh.models.Button(label="-", width=50)
    # compact_minus.on_click(time_controls.on_minus)
    compact_plus = bokeh.models.Button(label="+", width=50)
    # compact_plus.on_click(time_controls.on_plus)
    compact_navigation = bokeh.layouts.column(
            compact_button,
            bokeh.layouts.row(
                compact_minus,
                compact_plus,
                width=100))
    control_root = bokeh.layouts.column(
            compact_button,
            tabs,
            name="controls")

    display = "large"
    def on_compact():
        nonlocal display
        if display == "large":
            control_root.height = 100
            control_root.width = 120
            compact_button.width = 100
            compact_button.label = "Expand"
            control_root.children = [
                    compact_navigation]
            display = "compact"
        else:
            control_root.height = 500
            control_root.width = 300
            compact_button.width = 300
            compact_button.label = "Compact"
            control_root.children = [compact_button, tabs]
            display = "large"

    compact_button.on_click(on_compact)

    document = bokeh.plotting.curdoc()
    document.title = "FOREST"
    document.add_root(control_root)
    document.add_root(series_row)
    document.add_root(figure_row)


from itertools import cycle


class Series(object):
    def __init__(self, figure):
        self.figure = figure
        self.sources = {}
        circles = []
        items = []
        colors = cycle(bokeh.palettes.Colorblind[6][::-1])
        for name, loader in data.LOADERS.items():
            if not isinstance(loader, data.UMLoader):
                continue
            source = bokeh.models.ColumnDataSource({
                "x": [],
                "y": [],
            })
            color = next(colors)
            r = self.figure.line(
                    x="x",
                    y="y",
                    color=color,
                    line_width=1.5,
                    source=source)
            r.nonselection_glyph = bokeh.models.Line(
                    line_width=1.5,
                    line_color=color)
            c = self.figure.circle(
                    x="x",
                    y="y",
                    color=color,
                    source=source)
            c.selection_glyph = bokeh.models.Circle(
                    fill_color="red")
            c.nonselection_glyph = bokeh.models.Circle(
                    fill_color=color,
                    fill_alpha=0.5,
                    line_alpha=0)
            circles.append(c)
            items.append((name, [r]))
            self.sources[name] = source

        legend = bokeh.models.Legend(items=items,
                orientation="horizontal",
                click_policy="hide")
        self.figure.add_layout(legend, "below")

        tool = bokeh.models.HoverTool(
                tooltips=[
                    ('Time', '@x{%F %H:%M}'),
                    ('Value', '@y')
                ],
                formatters={
                    'x': 'datetime'
                })
        self.figure.add_tools(tool)

        tool = bokeh.models.TapTool(
                renderers=circles)
        self.figure.add_tools(tool)

        # Underlying state
        self.x = None
        self.y = None
        self.variable = None

    def on_field(self, action):
        variable, pressure, itime = action
        self.variable = variable
        self.pressure = pressure
        self.itime = itime
        self.render()

    def on_tap(self, event):
        self.x = event.x
        self.y = event.y
        self.render()

    def render(self):
        if any_none(self, ["x", "y", "variable"]):
            return
        self.figure.title.text = self.variable
        for name, source in self.sources.items():
            loader = data.LOADERS[name]
            source.data = loader.series(
                    self.variable,
                    self.x,
                    self.y,
                    self.pressure)
            source.selected.indices = [self.itime]


def any_none(obj, attrs):
    return any([getattr(obj, x) is None for x in attrs])


class Artist(object):
    def __init__(self, figures, color_mapper):
        self.figures = figures
        self.color_mapper = color_mapper
        self.viewers = {}
        self.renderers = {}
        self.previous_state = None
        self.variable = None
        self.pressure = None
        self.itime = 0
        for name, loader in data.LOADERS.items():
            if isinstance(loader, rdt.Loader):
                viewer = rdt.View(loader)
            elif isinstance(loader, earth_networks.Loader):
                viewer = earth_networks.View(loader)
            elif isinstance(loader, data.GPM):
                viewer = view.GPMView(loader, self.color_mapper)
            elif isinstance(loader, satellite.EIDA50):
                viewer = view.EIDA50(loader, self.color_mapper)
            else:
                viewer = view.UMView(loader, self.color_mapper)
            self.viewers[name] = viewer
            self.renderers[name] = [
                    viewer.add_figure(f)
                    for f in self.figures]

    def on_visible(self, state):
        print("on_visible", state)
        if self.previous_state is not None:
             # Hide deselected states
             lost_items = (
                     set(self.flatten(self.previous_state)) -
                     set(self.flatten(state)))
             for key, i, _ in lost_items:
                 self.renderers[key][i].visible = False

        # Sync visible states with menu choices
        states = set(self.flatten(state))
        hidden = [(i, j) for i, j, v in states if not v]
        visible = [(i, j) for i, j, v in states if v]
        for i, j in hidden:
            self.renderers[i][j].visible = False
        for i, j in visible:
            self.renderers[i][j].visible = True

        self.previous_state = dict(state)
        self.render()

    @staticmethod
    def flatten(state):
        items = []
        for key, flags in state.items():
            items += [(key, i, f) for i, f in enumerate(flags)]
        return items

    def on_field(self, action):
        variable, pressure, itime = action
        self.variable = variable
        self.pressure = pressure
        self.itime = itime
        self.render()

    def on_time_step(self, time_step):
        self.time_step = time_step
        self.render()

    def render(self):
        if self.previous_state is None:
            return
        for name in self.previous_state:
            viewer = self.viewers[name]
            if isinstance(viewer, view.UMView):
                viewer.render(
                        self.variable,
                        self.pressure,
                        self.itime)
            elif isinstance(viewer, view.EIDA50):
                if self.time_step is None:
                    continue
                if self.time_step.valid is None:
                    continue
                viewer.image(self.time_step.valid)
            elif isinstance(viewer, (rdt.View, earth_networks.View)):
                if self.time_step is None:
                    continue
                if self.time_step.valid is None:
                    continue
                viewer.render(self.time_step.valid)


class FieldControls(Observable):
    def __init__(self):
        self.variable = None
        self.itime = 0
        self.pressure = None
        super().__init__()

    def on_variable(self, value):
        self.variable = value
        self.render()

    def render(self):
        if self.variable is None:
            return
        self.announce(
                (self.variable,
                self.pressure,
                self.itime))

    def on_pressure_control(self, action):
        self.pressure = action
        self.render()

    def on_run_control(self, initial_time):
        print("run: {}".format(initial_time))
        self.initial_time = initial_time

    def on_time_control(self, action):
        itime, _ = action
        self.itime = itime
        self.render()


Timestep = namedtuple("Timestep", ("index", "length", "valid", "initial"))


class TimeControls(Observable):
    def __init__(self, steps):
        self.steps = steps
        self.labels = ["T{:+}".format(int(s))
                for s in self.steps]
        self.plus = bokeh.models.Button(label="+", width=80)
        self.plus.on_click(self.on_plus)
        self.minus = bokeh.models.Button(label="-", width=80)
        self.minus.on_click(self.on_minus)
        self.dropdown = bokeh.models.Dropdown(
                label="Time step",
                menu=list(zip(self.labels, self.labels)),
                width=80)
        self.dropdown.on_click(select(self.dropdown))
        self.dropdown.on_click(self.on_dropdown)
        sizing_mode = "fixed"
        self.layout = bokeh.layouts.row(
                bokeh.layouts.column(self.minus, width=90,
                    sizing_mode=sizing_mode),
                bokeh.layouts.column(self.dropdown, width=100,
                    sizing_mode=sizing_mode),
                bokeh.layouts.column(self.plus, width=90,
                    sizing_mode=sizing_mode),
                width=300)
        super().__init__()

    def set_times(self, times):
        self.steps = self.as_steps(times)
        self.labels = ["T{:+}".format(int(s))
                for s in self.steps]
        self.dropdown.menu = list(zip(
            self.labels, self.labels))

    @staticmethod
    def as_steps(times):
        t0 = times[0]
        return [(t - t0).total_seconds() / (60 * 60)
                for t in times]

    def on_plus(self):
        if self.dropdown.value is None:
            self.dropdown.value = self.labels[0]
            return
        if self.index == (len(self.labels) - 1):
            return
        else:
            value = self.labels[self.index + 1]
            self.dropdown.value = value

    def on_minus(self):
        if self.dropdown.value is None:
            self.dropdown.value = self.labels[0]
            return
        if self.index == 0:
            return
        else:
            value = self.labels[self.index - 1]
            self.dropdown.value = value

    def on_dropdown(self, value):
        self.announce((self.index, self.step))

    @property
    def index(self):
        if self.dropdown.value is None:
            return
        return self.labels.index(self.dropdown.value)

    @property
    def step(self):
        if self.index is None:
            return
        return self.steps[self.index]


class MapperLimits(object):
    def __init__(self, sources, color_mapper, fixed=False):
        self.fixed = fixed
        self.sources = sources
        for source in self.sources:
            source.on_change("data", self.on_source_change)
        self.color_mapper = color_mapper
        self.low_input = bokeh.models.TextInput(title="Low:")
        self.low_input.on_change("value",
                self.change(color_mapper, "low", float))
        self.color_mapper.on_change("low",
                self.change(self.low_input, "value", str))
        self.high_input = bokeh.models.TextInput(title="High:")
        self.high_input.on_change("value",
                self.change(color_mapper, "high", float))
        self.color_mapper.on_change("high",
                self.change(self.high_input, "value", str))
        self.checkbox = bokeh.models.CheckboxGroup(
                labels=["Fixed"],
                active=[])
        self.checkbox.on_change("active", self.on_checkbox_change)

    def on_checkbox_change(self, attr, old, new):
        if len(new) == 1:
            self.fixed = True
        else:
            self.fixed = False

    def on_source_change(self, attr, old, new):
        if self.fixed:
            return
        images = []
        for source in self.sources:
            if len(source.data["image"]) == 0:
                continue
            images.append(source.data["image"][0])
        if len(images) > 0:
            low = np.min([np.min(x) for x in images])
            high = np.max([np.max(x) for x in images])
            self.color_mapper.low = low
            self.color_mapper.high = high

    @staticmethod
    def change(widget, prop, dtype):
        def wrapper(attr, old, new):
            if old == new:
                return
            if getattr(widget, prop) == dtype(new):
                return
            setattr(widget, prop, dtype(new))
        return wrapper


def change(widget, prop, dtype):
    def wrapper(attr, old, new):
        if old == new:
            return
        if getattr(widget, prop) == dtype(new):
            return
        setattr(widget, prop, dtype(new))
    return wrapper


def add_feature(figure, data, color="black"):
    source = bokeh.models.ColumnDataSource(data)
    return figure.multi_line(
        xs="xs",
        ys="ys",
        source=source,
        color=color)


if __name__.startswith("bk"):
    main()
