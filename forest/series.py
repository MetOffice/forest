from itertools import cycle
import bokeh.palettes


class SeriesView(object):
    def __init__(self, figure, loaders):
        self.figure = figure
        self.loaders = loaders
        self.sources = {}
        circles = []
        items = []
        colors = cycle(bokeh.palettes.Colorblind[6][::-1])
        for name in self.loaders.keys():
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
        self.state = {}

    @classmethod
    def from_groups(cls, figure, groups, directory=None):
        loaders = {}
        for group in groups:
            if group.file_type == "unified_model":
                if directory is None:
                    pattern = group.full_pattern
                else:
                    pattern = os.path.join(directory, group.full_pattern)
                loaders[group.label] = data.SeriesLoader.from_pattern(pattern)
        return cls(figure, loaders)

    def on_state(self, app_state):
        next_state = dict(self.state)
        attrs = [
                "initial_time",
                "variable",
                "pressure"]
        for attr in attrs:
            if getattr(app_state, attr) is not None:
                next_state[attr] = getattr(app_state, attr)
        state_change = any(
                next_state.get(k, None) != self.state.get(k, None)
                for k in attrs)
        if state_change:
            self.render()
        self.state = next_state

    def on_tap(self, event):
        self.state["x"] = event.x
        self.state["y"] = event.y
        self.render()

    def render(self):
        for attr in ["x", "y", "variable", "initial_time"]:
            if attr not in self.state:
                return
        x = self.state["x"]
        y = self.state["y"]
        variable = self.state["variable"]
        initial_time = dt.datetime.strptime(
                self.state["initial_time"],
                "%Y-%m-%d %H:%M:%S")
        pressure = self.state.get("pressure", None)
        self.figure.title.text = variable
        for name, source in self.sources.items():
            loader = self.loaders[name]
            lon, lat = geo.plate_carree(x, y)
            lon, lat = lon[0], lat[0]  # Map to scalar
            source.data = loader.series(
                    initial_time,
                    variable,
                    lon,
                    lat,
                    pressure)
