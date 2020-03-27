import bokeh.models
import netCDF4
import forest.geo

class Dataset:
    def __init__(self, pattern=None, **kwargs):
        self.pattern = pattern

    def navigator(self):
        return Navigator()

    def map_view(self):
        return MapView(self.pattern)

    def profile_view(self, figure):
        return ProfileView(figure)


class Navigator:
    def initial_times(self, *args, **kwargs):
        return []

    def valid_times(self, *args, **kwargs):
        return []

    def variables(self, *args, **kwargs):
        return []

    def pressures(self, *args, **kwargs):
        return []


class MapView:
    def __init__(self, path):
        self.path = path
        with netCDF4.Dataset(self.path) as dataset:
            self.lons = dataset.variables["LONGITUDE"][:]
            self.lats = dataset.variables["LATITUDE"][:]
        self.source = bokeh.models.ColumnDataSource({
            "x": [],
            "y": [],
        })

    def add_figure(self, figure):
        return figure.circle(x="x", y="y", source=self.source)

    def render(self, *args, **kwargs):
        x, y = forest.geo.web_mercator(self.lons, self.lats)
        print(x, y)
        self.source.data = {
            "x": x,
            "y": y,
        }


class ProfileView:
    def __init__(self, figure):
        self.figure = figure
        self.source = bokeh.models.ColumnDataSource({
            "x": [],
            "y": [],
        })
        self.figure.circle(x="x", y="y", source=self.source)

    def connect(self, store):
        store.add_subscriber(self.render)

    def render(self, state):
        if "position" in state:
            x = state["position"]["x"]
            y = state["position"]["y"]
            lons, lats = forest.geo.plate_carree([x], [y])

            # Read data from file??
            self.source.stream({
                "x": lons,
                "y": lats,
            })
