from enum import Enum
import bokeh.plotting
import bokeh.events
import numpy as np
import os
import data
import view
import images
import geo
from util import Observable, select


class Mode(Enum):
    TIMESTEP = 0
    RUN = 1

NEXT = "NEXT"
PREVIOUS = "PREVIOUS"


def main():
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
        if isinstance(viewer, (view.UMView, view.GPMView)):
            image_sources.append(viewer.source)

    image_loaders = []
    for name, loader in data.LOADERS.items():
        if isinstance(loader, (data.UMLoader, data.GPM)):
            image_loaders.append(loader)

    features = []
    for figure in figures:
        features += [
            add_feature(figure, data.COASTLINES),
            add_feature(figure, data.BORDERS)]
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
    dropdown.on_click(change_label(dropdown))

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

    palettes = {
            "Viridis": bokeh.palettes.Viridis[256],
            "Magma": bokeh.palettes.Magma[256],
            "Inferno": bokeh.palettes.Inferno[256],
            "Plasma": bokeh.palettes.Plasma[256]
    }
    palette_controls = PaletteControls(
            color_mapper,
            palettes)

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

    variables = image_loaders[0].variables
    pressures = image_loaders[0].pressures
    pressure_variables = image_loaders[0].pressure_variables
    field_controls = FieldControls(
            variables,
            pressures,
            pressure_variables)

    image_controls.subscribe(artist.on_visible)
    field_controls.subscribe(artist.on_field)

    div = bokeh.models.Div(text="", width=10)
    border_row = bokeh.layouts.row(
        bokeh.layouts.column(toggle),
        bokeh.layouts.column(div),
        bokeh.layouts.column(dropdown))

    time_controls = TimeControls()
    time_controls.subscribe(field_controls.on_time_control)

    tabs = bokeh.models.Tabs(tabs=[
        bokeh.models.Panel(
            child=bokeh.layouts.column(
                time_controls.layout,
                bokeh.layouts.row(field_controls.drop),
                bokeh.layouts.row(field_controls.radio),
                image_controls.column),
            title="Data"),
        bokeh.models.Panel(
            child=bokeh.layouts.column(
                bokeh.layouts.row(figure_drop),
                border_row,
                bokeh.layouts.row(slider),
                bokeh.layouts.row(palette_controls.drop),
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

    document = bokeh.plotting.curdoc()
    document.add_root(
        bokeh.layouts.column(
            tabs,
            name="controls"))
    document.add_root(series_row)
    document.add_root(figure_row)


from itertools import cycle


class Series(object):
    def __init__(self, figure):
        self.figure = figure
        self.sources = {}
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
            items.append((name, [r]))
            self.sources[name] = source

        legend = bokeh.models.Legend(items=items,
                orientation="horizontal",
                click_policy="hide")
        self.figure.add_layout(legend, "below")

        # Underlying state
        self.x = None
        self.y = None
        self.variable = None
        self.ipressure = 0

    def on_field(self, variable, ipressure, itime):
        self.variable = variable
        self.ipressure = ipressure
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
            source.data = loader.series(self.variable, self.x, self.y, self.ipressure)


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
        self.ipressure = 0
        self.itime = 0
        for name, loader in data.LOADERS.items():
            if isinstance(loader, data.RDT):
                viewer = view.RDT(loader)
            elif isinstance(loader, data.EarthNetworks):
                viewer = view.EarthNetworks(loader)
            elif isinstance(loader, data.GPM):
                viewer = view.GPMView(loader, self.color_mapper)
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

    def on_field(self, variable, ipressure, itime):
        self.variable = variable
        self.ipressure = ipressure
        self.itime = itime
        self.render()

    def render(self):
        if self.previous_state is None:
            return
        for name in self.previous_state:
            viewer = self.viewers[name]
            if isinstance(viewer, view.UMView):
                viewer.render(self.variable, self.ipressure, self.itime)


class FieldControls(Observable):
    def __init__(self, variables, pressures, pressure_variables):
        self.variable = None
        self.mode = Mode.TIMESTEP
        self.itime = 0
        self.ipressure = 0
        self.variables = variables
        self.pressures = pressures
        self.pressure_variables = pressure_variables
        self.drop = bokeh.models.Dropdown(
                label="Variables",
                menu=[(v, v) for v in self.variables],
                width=50)
        self.drop.on_click(change_label(self.drop))
        self.drop.on_click(self.on_click)
        self.radio = bokeh.models.RadioGroup(
                labels=[str(int(p)) for p in self.pressures],
                active=self.ipressure,
                inline=True)
        self.radio.on_change("active", self.on_radio)
        super().__init__()

    def on_click(self, value):
        self.variable = value
        self.render()

    def on_radio(self, attr, old, new):
        if new is not None:
            self.ipressure = new
            self.render()

    def render(self):
        if self.variable is None:
            return
        if self.variable in self.pressure_variables:
            self.radio.disabled = False
        else:
            self.radio.disabled = True
        self.announce(self.variable, self.ipressure, self.itime)

    def on_time_control(self, action):
        if action == NEXT:
            self.next_time()
        elif action == PREVIOUS:
            self.previous_time()
        elif isinstance(action, Mode):
            self.mode = action

    def next_time(self):
        self.itime += +1
        self.render()

    def previous_time(self):
        self.itime += -1
        self.render()


class TimeControls(Observable):
    def __init__(self):
        self.plus = bokeh.models.Button(label="+", width=80)
        self.plus.on_click(self.on_plus)
        self.minus = bokeh.models.Button(label="-", width=80)
        self.minus.on_click(self.on_minus)
        self.modes = bokeh.models.Dropdown(
                label="Time step",
                menu=[("Time step", "Time step"), ("Model run", "Model run")],
                width=80)
        self.modes.on_click(select(self.modes))
        self.modes.on_click(self.on_mode)
        sizing_mode = "fixed"
        self.layout = bokeh.layouts.row(
                bokeh.layouts.column(self.minus, width=90,
                    sizing_mode=sizing_mode),
                bokeh.layouts.column(self.modes, width=100,
                    sizing_mode=sizing_mode),
                bokeh.layouts.column(self.plus, width=90,
                    sizing_mode=sizing_mode),
                width=300)
        super().__init__()

    def on_plus(self):
        self.announce(NEXT)

    def on_minus(self):
        self.announce(PREVIOUS)

    def on_mode(self, value):
        if value == "Time step":
            self.announce(Mode.TIMESTEP)
        else:
            self.announce(Mode.RUN)


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


class PaletteControls(object):
    def __init__(self, color_mapper, palettes):
        self.color_mapper = color_mapper
        self.palettes = palettes
        self.drop = bokeh.models.Dropdown(
                label="Palettes",
                menu=[(k, k) for k in self.palettes.keys()])
        self.drop.on_click(change_label(self.drop))
        self.drop.on_click(self.on_click)

    def on_click(self, value):
        self.color_mapper.palette = self.palettes[value]


def change_label(widget):
    def wrapped(value):
        widget.label = str(value)
    return wrapped


def change(widget, prop, dtype):
    def wrapper(attr, old, new):
        if old == new:
            return
        if getattr(widget, prop) == dtype(new):
            return
        setattr(widget, prop, dtype(new))
    return wrapper


def add_feature(figure, data):
    source = bokeh.models.ColumnDataSource(data)
    return figure.multi_line(
        xs="xs",
        ys="ys",
        source=source,
        color="white")


if __name__.startswith("bk"):
    main()
