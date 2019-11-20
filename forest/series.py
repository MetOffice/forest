import datetime as dt
from itertools import cycle
from collections import defaultdict
import bokeh.palettes
import numpy as np
import netCDF4
from forest import geo
from forest.observe import Observable
from forest.util import initial_time as _initial_time


SET_POSITION = "SET_POSITION"


def set_position(x, y):
    return {"kind": SET_POSITION, "payload": {"x": x, "y": y}}


def reducer(state, action):
    if action["kind"] == SET_POSITION:
        state["position"] = action["payload"]
    return state


class SeriesView(Observable):
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

        super().__init__()

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
            self.render(self.state)
        self.state = next_state

    def on_tap(self, event):
        self.notify(set_position(event.x, event.y))

    def render(self, state):
        for attr in ["x", "y", "variable", "initial_time"]:
            if attr not in state:
                return
        x = state["x"]
        y = state["y"]
        variable = state["variable"]
        initial_time = dt.datetime.strptime(
                state["initial_time"],
                "%Y-%m-%d %H:%M:%S")
        pressure = state.get("pressure", None)
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


class SeriesLoader(object):
    def __init__(self, paths):
        self.locator = SeriesLocator(paths)

    @classmethod
    def from_pattern(cls, pattern):
        return cls(sorted(glob.glob(os.path.expanduser(pattern))))

    def series(self,
            initial_time,
            variable,
            lon0,
            lat0,
            pressure=None):
        data = {"x": [], "y": []}
        paths = self.locator.locate(initial_time)
        for path in paths:
            segment = self.series_file(
                    path,
                    variable,
                    lon0,
                    lat0,
                    pressure=pressure)
            data["x"] += list(segment["x"])
            data["y"] += list(segment["y"])
        return data

    def series_file(self, *args, **kwargs):
        try:
            return self._load_netcdf4(*args, **kwargs)
        except:
            return self._load_cube(*args, **kwargs)

    def _load_cube(self, path, variable, lon0, lat0, pressure=None):
        """ Constrain data loading to points required """
        import iris
        cube = iris.load_cube(path, variable)
        # reference longitude axis by "axis='X'" and latitude axis as axis='Y',
        # to accommodate various types of coordinate system.
        # e.g. 'grid_longitude'. See iris.utils.guess_coord_axis.
        if cube.coord(axis='X').points[-1] > 180.0:
            # get circular longitude values
            lon0 = iris.analysis.cartography.wrap_lons(np.asarray(lon0), 0, 360)
        # Construct constraint
        coord_values={cube.coord(axis='X').standard_name: lon0,
                      cube.coord(axis='Y').standard_name: lat0,                     
                      }
        if pressure is not None and 'pressure' in [coord.name() for coord in cube.coords()]:
            ptol = 0.01 * pressure
            coord_values['pressure'] = (
                lambda cell: (pressure - ptol) < cell < (pressure + ptol)
            )
        cube = cube.extract(iris.Constraint(coord_values=coord_values))
        assert cube is not None
        # Get validity times and data values
        # list the validity times as datetime objects
        time_coord = cube.coord('time')
        times = time_coord.units.num2date(time_coord.points).tolist()
        values = cube.data
        return {
            "x": times,
            "y": values}

    def _load_netcdf4(self, path, variable, lon0, lat0, pressure=None):
        with netCDF4.Dataset(path) as dataset:
            try:
                var = dataset.variables[variable]
            except KeyError:
                return {"x": [], "y": []}
            lons = geo.to_180(self._longitudes(dataset, var))
            lats = self._latitudes(dataset, var)
            i = np.argmin(np.abs(lons - lon0))
            j = np.argmin(np.abs(lats - lat0))
            times = self._times(dataset, var)
            values = var[..., j, i]
            if (
                    ("pressure" in var.coordinates) or
                    ("pressure" in var.dimensions)):
                pressures = self._pressures(dataset, var)
                if len(var.dimensions) == 3:
                    pts = self.search(pressures, pressure)
                    values = values[pts]
                    if isinstance(times, dt.datetime):
                        times = [times]
                    else:
                        times = times[pts]
                else:
                    mask = self.search(pressures, pressure)
                    values = values[:, mask][:, 0]
        return {
            "x": times,
            "y": values}

    @staticmethod
    def _times(dataset, variable):
        """Find times related to variable in dataset"""
        time_dimension = variable.dimensions[0]
        coordinates = variable.coordinates.split()
        for c in coordinates:
            if c.startswith("time"):
                try:
                    var = dataset.variables[c]
                    return netCDF4.num2date(var[:], units=var.units)
                except KeyError:
                    pass
        for v, var in dataset.variables.items():
            if len(var.dimensions) != 1:
                continue
            if v.startswith("time"):
                d = var.dimensions[0]
                if d == time_dimension:
                    return netCDF4.num2date(var[:], units=var.units)

    def _pressures(self, dataset, variable):
        return self._dimension("pressure", dataset, variable)

    def _longitudes(self, dataset, variable):
        return self._dimension("longitude", dataset, variable)

    def _latitudes(self, dataset, variable):
        return self._dimension("latitude", dataset, variable)

    @staticmethod
    def _dimension(prefix, dataset, variable):
        for d in variable.dimensions:
            if not d.startswith(prefix):
                continue
            if d in dataset.variables:
                return dataset.variables[d][:]
        for c in variable.coordinates.split():
            if not c.startswith(prefix):
                continue
            if c in dataset.variables:
                return dataset.variables[c][:]

    @staticmethod
    def search(pressures, pressure, rtol=0.01):
        return np.abs(pressures - pressure) < (rtol * pressure)


class SeriesLocator(object):
    """Helper to find files related to Series"""
    def __init__(self, paths):
        self.paths = paths
        self.table = defaultdict(list)
        for path in paths:
            time = _initial_time(path)
            if time is None:
                try:
                    with netCDF4.Dataset(path) as dataset:
                        var = dataset.variables["forecast_reference_time"]
                        time = netCDF4.num2date(var[:], units=var.units)
                except KeyError:
                    continue
            self.table[self.key(time)].append(path)

    def initial_times(self):
        return np.array(list(self.table.keys()),
                dtype='datetime64[s]')

    def locate(self, initial_time):
        if isinstance(initial_time, str):
            return self.table[initial_time]
        if isinstance(initial_time, np.datetime64):
            initial_time = initial_time.astype(dt.datetime)
        return self.table[self.key(initial_time)]

    __getitem__ = locate

    def key(self, time):
        return "{:%Y-%m-%d %H:%M:%S}".format(time)
