"""
Profile sub plot
----------------

Support for the profile window uses the redux
design pattern. The View reacts to State
changes.

.. autoclass:: ProfileView
    :members:

.. autoclass:: ProfileLoader
    :members:

.. autoclass:: ProfileLocator
    :members:

Reducer
~~~~~~~

A reducer is a pure function that combines a
state and an action to return a new state. Reducers
can be combined so that individual reducers are
responsible only for a limited set of actions

.. autofunction:: reducer

Selector
~~~~~~~~

The full application state can be unwieldy for individual
views to parse, especially if the details of its internals
change. Instead of relying on Views to translate state
into values, selectors can do that job on behalf of the view.

.. autofunction:: select_args

"""
import copy
import datetime as dt
import glob
import os
from itertools import cycle
from collections import defaultdict
import bokeh.palettes
import numpy as np
import netCDF4
from forest import geo
from forest.observe import Observable
from forest.redux import Action
from forest.util import initial_time as _initial_time
from forest.gridded_forecast import _to_datetime
from forest.screen import SET_POSITION
try:
    import iris
except ModuleNotFoundError:
    iris = None
    # ReadTheDocs can't import iris


def reducer(state, action):
    """Profile specific reducer

    Given :func:`screen.set_position` action adds "position" data
    to state

    :param state: data structure representing current state
    :type state: dict
    :param action: data structure representing action
    :type action: dict
    """
    state = copy.deepcopy(state)
    if action["kind"] == SET_POSITION:
        state["position"] = action["payload"]
    return state


def select_args(state):
    """Select args needed by :func:`ProfileView.render`

    .. note:: If all criteria are not present None is returned

    :returns: args tuple or None
    """
    if any(att not in state
            for att in [
                "variable",
                "initial_time",
                "position"]):
        return
    if "valid_time" in state:
        optional = (_to_datetime(state["valid_time"]),)
    else:
        optional = ()
    return (
            _to_datetime(state["initial_time"]),
            state["variable"],
            state["position"]["x"],
            state["position"]["y"],
            state["tools"]["profile"]) + optional

def _find_nearest(value, array):
    idx = (np.abs(array - value)).argmin()
    return array[np.array(idx)]


class ProfileView(Observable):
    """Profile view

    Responsible for keeping the lines on the profile figure
    up to date.
    """
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

        super().__init__()

    @classmethod
    def from_groups(cls, figure, groups):
        """Factory method to load from :class:`~forest.config.FileGroup` objects"""
        loaders = {}
        for group in groups:
            if group.file_type == "unified_model":
                pattern = group.full_pattern
                loaders[group.label] = ProfileLoader.from_pattern(pattern)
        return cls(figure, loaders)

    def render(self, initial_time, variable, x, y, visible, time=None):
        """Update data for a particular application setting"""
        if visible:
            assert isinstance(initial_time, dt.datetime), "only support datetime"
            self.figure.title.text = variable
            for name, source in self.sources.items():
                loader = self.loaders[name]
                lon, lat = geo.plate_carree(x, y)
                lon, lat = lon[0], lat[0]  # Map to scalar
                source.data = loader.profile(
                        initial_time,
                        variable,
                        lon,
                        lat,
                        time)


class ProfileLoader(object):
    """Profile loader"""
    def __init__(self, paths):
        self.locator = ProfileLocator(paths)

    @classmethod
    def from_pattern(cls, pattern):
        return cls(sorted(glob.glob(os.path.expanduser(pattern))))

    def profile(self,
            initial_time,
            variable,
            lon0,
            lat0,
            time=None):
        data = {"x": [], "y": []}
        paths = self.locator.locate(initial_time)
        for path in paths:
            segment = self.profile_file(
                    path,
                    variable,
                    lon0,
                    lat0,
                    time=time)
            data["x"] += list(segment["x"])
            data["y"] += list(segment["y"])
        return data

    def profile_file(self, *args, **kwargs):
        return self._load_cube(*args, **kwargs)

    def _load_cube(self, path, variable, lon0, lat0, time=None):
        """ Constrain data loading to points required """

        cube = iris.load_cube(path, variable)

        # reference longitude axis by "axis='X'" and latitude axis as axis='Y',
        # to accommodate various types of coordinate system.
        # e.g. 'grid_longitude'. See iris.utils.guess_coord_axis.
        if cube.coord(axis='X').points[-1] > 180.0:
            # get circular longitude values
            lon0 = iris.analysis.cartography.wrap_lons(np.asarray(lon0), 0, 360)
        # Construct constraint
        lon_nearest = _find_nearest(lon0, cube.coord(axis='X').points)
        lat_nearest = _find_nearest(lat0, cube.coord(axis='Y').points)
        coord_values={
            cube.coord(axis='X').standard_name: (lambda cell: lon_nearest == cell.point),
            cube.coord(axis='Y').standard_name: (lambda cell: lat_nearest == cell.point),
            }
        if time is not None and 'time' in [coord.name() for coord in cube.coords()]:
            coord_values['time'] = (
                lambda cell: _to_datetime(time) == cell
            )
        constraint = iris.Constraint(coord_values=coord_values)

        # Extract nearest profile
        cube = cube.extract(constraint)
        assert cube is not None, "Error: No profile data found for these coordinates"

        # Get level info and data values
        if 'pressure' in [coord.name() for coord in cube.coords()]: 
            pressure_coord = cube.coord('pressure')
            pressures = pressure_coord.points.tolist()
        else:
            pressures = [0,]
        values = cube.data
        return {
            "x": values,
            "y": pressures}


class ProfileLocator(object):
    """Helper to find files related to Profile"""
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
