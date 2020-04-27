from collections import defaultdict
try:
    import cartopy
except ImportError:
    # ReadTheDocs unable to pip install cartopy
    pass
import numpy as np
from forest import geo
try:
    import shapely.geometry
except ImportError:
    # ReadTheDocs unable to pip install shapely
    pass


# Application data shared across documents
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
FEATURE_FLAGS = defaultdict(lambda: False)

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
