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
import datetime as dt
import glob
import os
from itertools import cycle
import collections
from collections import defaultdict
import bokeh.palettes
import numpy as np
import netCDF4
try:
    import xarray
except ModuleNotFoundError:
    xarray = None
from forest import geo
from forest.observe import Observable
from forest.redux import Action
from forest.util import initial_time as _initial_time
from forest.util import to_datetime as _to_datetime
from forest.screen import SET_POSITION
try:
    import iris
except ModuleNotFoundError:
    iris = None
    # ReadTheDocs can't import iris

#Nanoseconds per second
NS = 1e-9

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
        legend.label_text_font_size = "8pt"
        self.figure.add_layout(legend, "below")

        tool = bokeh.models.HoverTool(
                tooltips=[
                    ('Value', '@x'),
                    ('Level', '@y')
                ])
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
        paths = self.locator.locate(initial_time, time)
        for path in paths:
            segment = self.profile_file(
                    path,
                    variable,
                    lon0,
                    lat0,
                    time)
            data["x"] += list(segment["x"])
            data["y"] += list(segment["y"])
        return data

    def profile_file(self, *args, **kwargs):
        """ Read profile data from a file."""
        return self._load_cube(*args, **kwargs)

    def _load_cube(self, path, variable, lon0, lat0, time=None):
        """ Load vertical profile slice from file via iris. """

        try:
            cube = iris.load_cube(path, variable)
        except iris.exceptions.ConstraintMismatchError as e:
            print("WARNING: {} No data found for profile plot".format(type(e).__name__))
            return {
                "x": [],
                "y": []}

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
        assert cube is not None, ("Error: No profile data found for {}\n\t"
                                  "at these coordinates: time,lat,lon {},{},{}").format(
                                  path, _to_datetime(time), lat_nearest, lon_nearest)

        # Get level info and data values
        if 'pressure' in [coord.name() for coord in cube.coords()]:
            pressure_coord = cube.coord('pressure')
            pressures = pressure_coord.points.tolist()
        else:
            pressures = [0,]
        if time is None and 'time' in [coord.name() for coord in cube.coords()]:
            print("Warning: no time specified, selecting first element of array")
            values = cube.data[0,...]
        else:
            values = cube.data

        return {
            "x": values,
            "y": pressures}


class ProfileLocator(object):
    """Helper to find files related to Profile"""
    def __init__(self, paths):
        self.paths = paths
        self.valid_times_to_paths = defaultdict(list)
        self._ini_times_to_paths = None

    @property
    def ini_times_to_paths(self):
        if self._ini_times_to_paths is None:
            self._ini_times_to_paths = self._map_ini_times_to_paths(self.paths)
        return self._ini_times_to_paths

    def _map_ini_times_to_paths(self, paths):
        """
        .. note:: Potentially expensive I/O operation
        """
        mapping = defaultdict(list)
        for path in paths:
            initial_time = _initial_time(path)
            try:
                with netCDF4.Dataset(path) as dataset:
                    if initial_time is None:
                        var = dataset.variables["forecast_reference_time"]
                        initial_time = netCDF4.num2date(var[:], units=var.units)
            except (FileNotFoundError, KeyError) as ex:
                pass
            mapping[self.key(initial_time)].append(path)
        return mapping

    def initial_times(self):
        return np.array(list(self.ini_times_to_paths.keys()),
                dtype='datetime64[s]')

    def locate(self, initial_time, valid_time=None):
        if isinstance(initial_time, str):
            initial_time_paths = self.ini_times_to_paths[initial_time]
        elif isinstance(initial_time, np.datetime64):
            initial_time = initial_time.astype(dt.datetime)
            initial_time_paths = self.ini_times_to_paths[self.key(initial_time)]
        else:
            initial_time_paths = self.ini_times_to_paths[self.key(initial_time)]
        if valid_time is None:
            return initial_time_paths
        elif xarray is None:
            return []  # Always return list
        else:
            # set valid times
            for path in initial_time_paths:
                with xarray.open_dataset(path) as xr_ds:
                    for name in xr_ds.coords:
                        if getattr(xr_ds.coords[name], "standard_name", None) == 'time':
                            times = xr_ds.coords[name].values
                            try:
                                for time in times:
                                    time = dt.datetime.utcfromtimestamp(time.astype(int)*NS)
                                    self.valid_times_to_paths[self.key(time)].append(path)
                            except TypeError:
                                time = dt.datetime.utcfromtimestamp(times.astype(int)*NS)
                                self.valid_times_to_paths[self.key(time)].append(path)
                            break
            valid_time_paths = self.valid_times_to_paths[self.key(valid_time)]
            return list(set(initial_time_paths).intersection(set(valid_time_paths)))

    def key(self, time):
        return "{:%Y-%m-%d %H:%M:%S}".format(time)
