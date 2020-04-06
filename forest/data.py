try:
    import cartopy
except ImportError:
    # ReadTheDocs unable to pip install cartopy
    pass
import numpy as np
import netCDF4
from forest import geo
import bokeh.models
from collections import OrderedDict
from functools import partial
try:
    import shapely.geometry
except ImportError:
    # ReadTheDocs unable to pip install shapely
    pass
import forest.util


# Application data shared across documents
VECTORS = OrderedDict()
COASTLINES = {
    "xs": [],
    "ys": []
}
BORDERS = {
    "xs": [],
    "ys": []
}
LAKES = {
    "xs": [],
    "ys": []
}
DISPUTED = {
    "xs": [],
    "ys": []
}
AUTO_SHUTDOWN = False

def on_server_loaded():
    global DISPUTED
    global COASTLINES
    global LAKES
    global BORDERS
    # Load coastlines/borders
    EXTENT = (-10, 50, -20, 10)
    COASTLINES = load_coastlines()
    LAKES = xs_ys(iterlines(
        cartopy.feature.NaturalEarthFeature(
            'physical',
            'lakes',
            '10m').intersecting_geometries(EXTENT)))
    DISPUTED = xs_ys(iterlines(
            cartopy.feature.NaturalEarthFeature(
                "cultural",
                "admin_0_boundary_lines_disputed_areas",
                "50m").geometries()))
    BORDERS = xs_ys(iterlines(
        cartopy.feature.NaturalEarthFeature(
            'cultural',
            'admin_0_boundary_lines_land',
            '50m').geometries()))


def load_coastlines():
    return xs_ys(cut(iterlines(
            cartopy.feature.COASTLINE.geometries()), 180))


def xs_ys(lines):
    """Map to Web Mercator projection and bokeh multi_line structure"""
    xs, ys = [], []
    for lons, lats in lines:
        x, y = geo.web_mercator(lons, lats)
        xs.append(x)
        ys.append(y)
    return {
        "xs": xs,
        "ys": ys
    }


def cut(lines, x):
    """Cut lines in two if they cross a vertical line"""
    for line in lines:
        xs, ys = line
        xs, ys = np.ma.asarray(xs), np.ma.asarray(ys)
        if (np.min(xs) < x) and (np.max(xs) > x):
            pts = np.ma.asarray(xs) < x
            yield xs[pts], ys[pts]
            yield xs[~pts], ys[~pts]
        else:
            yield line


def iterlines(geometries):
    """Iterate lines from cartopy geometry"""
    def xy(g):
        if isinstance(g, shapely.geometry.LineString):
            return g.xy
        else:
            return g.exterior.coords.xy
    for geometry in geometries:
        try:
            for g in geometry:
                yield xy(g)
        except TypeError:
            yield xy(geometry)


# TODO: Delete this decorator in a future PR
def cache(name):
    store = globals()[name]
    def decorator(f):
        def wrapped(*args):
            if args not in store:
                store[args] = f(*args)
            return store[args]
        return wrapped
    return decorator


# TODO: Delete this class in a future PR
class ActiveViewer(object):
    def __init__(self):
        self.active = False
        self.loaded_state = None
        self.pending_state = None

    def add_figure(self, figure):
        raise Exception("this method should be implemented")

    def connect_data(self, figure, renderer):
        self.active = True
        if self.pending_state is not None:
            self.load(self.pending_state)
            self.loaded_state = self.pending_state
            self.pending_state = None
        def hide(renderer):
            renderer.visible = False
            self.active = False
        return partial(hide, renderer)

    def on_state(self, state):
        if self.active:
            if (self.loaded_state != state):
                self.load(state)
                self.loaded_state = state
            self.pending_state = None
        else:
            self.pending_state = state


# TODO: Delete this class in a future PR
class WindBarbs(ActiveViewer):
    def __init__(self, paths):
        self.paths = paths
        self.finder = Finder(paths)
        self.source = bokeh.models.ColumnDataSource({
            "x": [],
            "y": [],
            "u": [],
            "v": []})
        super().__init__()

    def add_figure(self, figure):
        renderer = figure.barb(
                x="x",
                y="y",
                u="u",
                v="v",
                source=self.source)
        return super().connect_data(figure, renderer)

    def load(self, state):
        time_step, ipressure = state
        if time_step is None:
            return
        if ipressure is None:
            return
        path = self.find_path(time_step.initial)
        # itime = self.time_index(time_step.length)
        itime = time_step.index
        self.source.data = self.load_data(
                path,
                itime,
                ipressure)

    def find_path(self, initial_time):
        return self.finder.find(initial_time)

    def time_index(self, length):
        return 0

    @staticmethod
    @cache("VECTORS")
    def load_data(path, itime, ipressure):
        step = 100
        with netCDF4.Dataset(path) as dataset:
            var = dataset.variables["x_wind"]
            if len(var.dimensions) == 4:
                values = var[itime, ipressure]
            else:
                values = var[itime]
            u = values[::step, ::step]
            var = dataset.variables["y_wind"]
            if len(var.dimensions) == 4:
                values = var[itime, ipressure]
            else:
                values = var[itime]
            v = values[::step, ::step]
            for d in var.dimensions:
                if "longitude" in d:
                    lons = dataset.variables[d][::step]
                if "latitude" in d:
                    lats = dataset.variables[d][::step]
        gx, _ = geo.web_mercator(
            lons,
            np.zeros(len(lons), dtype="d"))
        _, gy = geo.web_mercator(
            np.zeros(len(lats), dtype="d"),
            lats)
        x, y = np.meshgrid(gx, gy)
        u = forest.util.convert_units(u, 'm s-1', 'knots')
        v = forest.util.convert_units(v, 'm s-1', 'knots')
        return {
            "x": x.flatten(),
            "y": y.flatten(),
            "u": u.flatten(),
            "v": v.flatten()
        }


# TODO: Delete this class in a future PR
class Finder(object):
    def __init__(self, paths):
        self.paths = paths
        self.table = {
                forest.util.initial_time(p): p for p in paths}
        self.initial_times = np.array(
                [forest.util.initial_time(p) for p in paths],
                dtype='datetime64[s]')
        with netCDF4.Dataset(self.paths[0]) as dataset:
            if "pressure" in dataset.variables:
                self.pressures = dataset.variables["pressure"][:]
            else:
                self.pressures = None
            self.pressure_variables = self.load_heights(dataset)

    @staticmethod
    def load_heights(dataset):
        variables = set()
        for variable, var in dataset.variables.items():
            if variable == "pressure":
                continue
            if "pressure" in var.dimensions:
                variables.add(variable)
        return variables

    def find(self, initial, pressure, variable):
        if variable in self.pressure_variables:
            ipressure = 0
        else:
            ipressure = self.pressure_index(
                    self.pressures,
                    pressure)
        return self.find_path(initial), ipressure

    def pressure_index(self, pressures, pressure):
        if isinstance(pressures, list):
            pressures = np.array(pressures, dtype="f")
        return np.argmin(np.abs(pressures - pressure))

    def find_path(self, initial_time):
        try:
            return self.table[initial_time]
        except KeyError:
            initial_time = np.datetime64(
                    initial_time, 's')
            i = np.argmin(
                    np.abs(
                        self.initial_times - initial_time))
            return self.paths[i]
