import copy
import bokeh.models
import numpy as np
import netCDF4
import forest.geo
from forest.redux import Action
from forest.observe import Observable
from screen import SET_PROFILE_IDS

def set_profile_ids(i, profile_ids) -> Action:
    """Action that stores selected profile_ids

    .. code-block:: python

        {
            "kind": "SET_PROFILE_IDS",
            "payload": {
                "indices": i,
                "profile_ids": profile_ids
            }
        }

    :returns: data representing action
    :rtype: dict
    """
    return {"kind": SET_PROFILE_IDS,
            "payload": {"indices": i, "ids": profile_ids}}

class Dataset:
    def __init__(self, pattern=None, **kwargs):
        self.pattern = pattern

    def navigator(self):
        return Navigator()

    def map_view(self):
        return MapView(self.pattern)

    def profile_view(self, figure):
        return ProfileView(self.pattern, figure)


class Navigator:
    def initial_times(self, *args, **kwargs):
        return []

    def valid_times(self, *args, **kwargs):
        return []

    def variables(self, *args, **kwargs):
        return []

    def pressures(self, *args, **kwargs):
        return []


class MapView(Observable):
    def __init__(self, path):
        self.path = path
        with netCDF4.Dataset(self.path) as dataset:
            self.lons = dataset.variables["LONGITUDE"][:]
            self.lats = dataset.variables["LATITUDE"][:]
        self.source = bokeh.models.ColumnDataSource({
            "x": [],
            "y": [],
        })
        # tap events never happen for some reason...
        self.source.on_event("tap", self.callback)
        # however selected.indices does change on the data source on a tap
        self.source.selected.on_change("indices", self.update_profile_ids)
        super().__init__()

    def callback(self, event):
        print("hit argo.MapView.callback: ", event)

    def update_profile_ids(self, a, o, n):
        print(a, o, n)
        with netCDF4.Dataset(self.path) as dataset:
            plat_ids = []
            for i in n:
                plat_id = dataset.variables["PLATFORM_NUMBER"][i,:]
                plat_ids.append(str(netCDF4.chartostring(plat_id)))

        self.notify(set_profile_ids(n, plat_ids))

    def add_figure(self, figure):
        # Tap event listener
        renderer = figure.circle(x="x", y="y", source=self.source)
        figure.add_tools(bokeh.models.TapTool(renderers=[renderer]))
        return renderer

    def render(self, *args, **kwargs):
        x, y = forest.geo.web_mercator(self.lons, self.lats)
        self.source.data = {
            "x": x,
            "y": y,
        }

    def connect(self, store):
        self.add_subscriber(store.dispatch)


class ProfileView:
    def __init__(self, path, figure):
        self.path = path
        self.figure = figure
        self.source = bokeh.models.ColumnDataSource({
            "x": [],
            "y": [],
        })
        self.figure.circle(x="x", y="y", source=self.source)

    def connect(self, store):
        store.add_subscriber(self.render)

    def render(self, state):
        if "profile_ids" in state:
            profile_index = state["profile_ids"]["indices"][0]
            print("render profile for platform ", state["profile_ids"]["ids"][0])

            # Read data from file, here for now, but might be better off
            # somewhere else.
            with netCDF4.Dataset(self.path) as dataset:
               temp  = dataset.variables["TEMP"][profile_index, :].data
               #pressure = dataset.variables["PRES"][profile_index, :].data
               pressure = np.r_[0:len(temp):1]

            self.source.data = {
                    "x": temp,
                    "y": pressure,
                }
