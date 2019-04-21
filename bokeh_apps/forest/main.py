import bokeh.plotting
import bokeh.events
import numpy as np
import os
import data
import view
import images
import geo
import picker
import colors
from util import Observable, select
from collections import defaultdict
import datetime as dt


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

    variables = image_loaders[0].variables
    pressures = image_loaders[0].pressures
    field_controls = FieldControls(variables)
    image_controls.subscribe(artist.on_visible)
    field_controls.subscribe(artist.on_field)

    div = bokeh.models.Div(text="", width=10)
    border_row = bokeh.layouts.row(
        bokeh.layouts.column(toggle),
        bokeh.layouts.column(div),
        bokeh.layouts.column(dropdown))

    name = list(data.FILE_DB.files.keys())[0]
    paths = data.FILE_DB.files[name]
    run_controls = RunControls.paths(paths)
    run_controls.subscribe(
            field_controls.on_run_control)

    loader = data.LOADERS[name]
    time_controls = TimeControls.times(
            loader.times["time"])
    time_controls.subscribe(field_controls.on_time_control)
    time_controls.subscribe(run_controls.on_step)

    pressure_controls = PressureControls(
            pressures,
            units="hPa")
    pressure_controls.subscribe(field_controls.on_pressure_control)

    tabs = bokeh.models.Tabs(tabs=[
        bokeh.models.Panel(
            child=bokeh.layouts.column(
                run_controls.layout,
                time_controls.layout,
                pressure_controls.layout,
                bokeh.layouts.row(field_controls.drop)),
            title="Navigate"),
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
    compact_minus.on_click(time_controls.on_minus)
    compact_plus = bokeh.models.Button(label="+", width=50)
    compact_plus.on_click(time_controls.on_plus)
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
        self.ipressure = 0

    def on_field(self, variable, ipressure, itime):
        self.variable = variable
        self.ipressure = ipressure
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
            source.data = loader.series(self.variable, self.x, self.y, self.ipressure)
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


class PressureControls(Observable):
    def __init__(self, pressures, units):
        self.pressures = pressures
        self.units = units
        self.labels = ["{} {}".format(int(p), self.units)
                for p in self.pressures]
        menu = list(zip(self.labels, self.labels))
        self.dropdown = bokeh.models.Dropdown(
                label="Pressures",
                menu=menu,
                width=80)
        self.dropdown.on_click(select(self.dropdown))
        self.dropdown.on_click(self.on_dropdown)
        self.plus = bokeh.models.Button(
                label="+", width=80)
        self.plus.on_click(self.on_plus)
        self.minus = bokeh.models.Button(
                label="-", width=80)
        self.minus.on_click(self.on_minus)
        sizing_mode = "fixed"
        self.layout = bokeh.layouts.row(
                bokeh.layouts.column(
                    self.minus, width=90,
                    sizing_mode=sizing_mode),
                bokeh.layouts.column(
                    self.dropdown, width=100,
                    sizing_mode=sizing_mode),
                bokeh.layouts.column(
                    self.plus, width=90,
                    sizing_mode=sizing_mode),
                width=300)
        super().__init__()

    def on_dropdown(self, value):
        self.announce(self.index)

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
            self.dropdown.value = self.labels[-1]
            return
        if self.index == 0:
            return
        else:
            value = self.labels[self.index - 1]
            self.dropdown.value = value
            self.announce(self.index)

    @property
    def index(self):
        if self.dropdown.value is None:
            return
        return self.labels.index(self.dropdown.value)


class FieldControls(Observable):
    def __init__(self, variables):
        self.variable = None
        self.itime = 0
        self.ipressure = 0
        self.variables = variables
        self.drop = bokeh.models.Dropdown(
                label="Variables",
                menu=[(v, v) for v in self.variables],
                width=50)
        self.drop.on_click(select(self.drop))
        self.drop.on_click(self.on_click)
        super().__init__()

    def on_click(self, value):
        self.variable = value
        self.render()

    def render(self):
        if self.variable is None:
            return
        self.announce(
                self.variable,
                self.ipressure,
                self.itime)

    def on_pressure_control(self, ipressure):
        self.ipressure = ipressure
        self.render()

    def on_run_control(self, initial_time):
        print("run: {}".format(initial_time))

    def on_time_control(self, action):
        itime, _ = action
        self.itime = itime
        self.render()


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

    @classmethod
    def times(cls, times):
        return cls(cls.as_steps(times))

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
            self.announce((self.index, self.step))

    def on_minus(self):
        if self.dropdown.value is None:
            self.dropdown.value = self.labels[0]
            return
        if self.index == 0:
            return
        else:
            value = self.labels[self.index - 1]
            self.dropdown.value = value
            self.announce((self.index, self.step))

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


class RunControls(Observable):
    def __init__(self, initial_times):
        # Internal state
        self.date = None
        self.time = None
        self.initial_times = initial_times
        self._times = defaultdict(set)
        for datetime in self.initial_times:
            d = self.as_date(datetime)
            t = self.as_time(datetime)
            self._times[d].add(t)

        # Widget(s)
        width = 80
        side = 90
        middle = 100
        row = 300
        sizing_mode = 'fixed'
        plus = bokeh.models.Button(
                label="+",
                width=width)
        plus.on_click(self.on_plus)
        minus = bokeh.models.Button(
                label="-",
                width=width)
        minus.on_click(self.on_minus)
        self.dropdown = bokeh.models.Dropdown(
                label="Initial time",
                width=width)
        self.dropdown.on_click(select(self.dropdown))
        self.dropdown.on_click(self.on_time)
        self.valid_div = bokeh.models.Div(
                text="Valid date",
                width=row - 30)
        self.date_picker = picker.DayPicker(
                title="Initial date",
                width=row - 30)
        self.date_picker.on_change('value', self.on_date)
        button_row = bokeh.layouts.row(
                bokeh.layouts.column(
                    minus,
                    width=side,
                    sizing_mode=sizing_mode),
                bokeh.layouts.column(
                    self.dropdown,
                    width=middle,
                    sizing_mode=sizing_mode),
                bokeh.layouts.column(
                    plus,
                    width=side,
                    sizing_mode=sizing_mode),
                width=row)
        self.layout = bokeh.layouts.column(
                bokeh.layouts.row(
                    self.valid_div,
                    width=row),
                bokeh.layouts.row(
                    self.date_picker,
                    width=row),
                button_row)

        # Observable
        super().__init__()

    def latest(self):
        print("Select latest model run")

    @classmethod
    def paths(cls, paths):
        return cls([data.initial_time(p) for p in paths])

    @property
    def initial_dates(self):
        return [self.as_date(t)
                for t in self.initial_times]

    def times(self, date):
        return sorted(self._times[date])

    @staticmethod
    def as_date(t):
        return dt.date(t.year, t.month, t.day)

    @staticmethod
    def as_time(t):
        return dt.time(t.hour, t.minute, t.second)

    def on_plus(self):
        if self.initial_time is None:
            self.initial_time = self.initial_times[0]
            self.date_picker.value = self.date
            self.dropdown.value = "{:%H:%M}".format(self.initial_time)

    def on_minus(self):
        print('minus')

    def on_date(self, attr, old, new):
        self.date = new
        if self.initial_time is not None:
            self.announce(self.initial_time)

        if new not in self._times:
            self.dropdown.disabled = True
        else:
            times = self._times[new]
            strings = ["{:%H:%M}".format(t) for t in times]
            menu = [(s, s) for s in strings]
            self.dropdown.disabled = False
            self.dropdown.menu = menu

    def on_step(self, action):
        _, hours = action
        print(hours)
        valid_date = self.initial_time + dt.timedelta(hours=int(hours))
        self.valid_div.text = "Valid date<br><span style='margin-left:10px'><b>{:%a %b %d %Y %H:%M}</b></span>".format(valid_date)

    def on_time(self, value):
        self.time = dt.datetime.strptime(
                value, "%H:%M")
        if self.initial_time is not None:
            self.announce(self.initial_time)

    @property
    def initial_time(self):
        if self.date is None:
            return
        if self.time is None:
            return
        return dt.datetime(
                self.date.year,
                self.date.month,
                self.date.day,
                self.time.hour,
                self.time.minute)

    @initial_time.setter
    def initial_time(self, value):
        self.date = self.as_date(value)
        self.time = self.as_time(value)


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
