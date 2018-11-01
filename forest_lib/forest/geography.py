"""Geographical features, coast lines and political boundaries"""
import bokeh.models
import numpy as np
import cartopy

__all__ = [
    "bounding_square",
    "add_coastlines",
    "add_borders",
    "coastlines"
]


def bounding_square(x0, y0, x1, y1):
    """Estimate bounding square given rectangle"""
    dx = x1 - x0
    dy = y1 - y0
    if dx > dy:
        side = dx
    else:
        side = dy
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    rx0 = cx - (side / 2)
    ry0 = cy - (side / 2)
    rx1 = cx + (side / 2)
    ry1 = cy + (side / 2)
    return rx0, ry0, rx1, ry1


def add_coastlines(figure, scale="50m"):
    feature = cartopy.feature.COASTLINE
    feature.scale = scale
    return PeriodicArtist(figure, feature)


def add_borders(figure, scale="50m"):
    feature = cartopy.feature.BORDERS
    feature.scale = scale
    return PeriodicArtist(figure, feature)


class PeriodicArtist(object):
    def __init__(self, figure, feature, interval=500):
        self.pcb = None
        self.figure = figure
        self.feature = feature
        self.interval = interval
        self.source = bokeh.models.ColumnDataSource({
            "xs": [],
            "ys": []
        })
        figure.multi_line(xs="xs", ys="ys",
                          source=self.source,
                          color="black")
        figure.x_range.range_padding = 0.
        figure.y_range.range_padding = 0.
        figure.x_range.on_change("start", self.on_change)
        figure.x_range.on_change("end", self.on_change)
        figure.x_range.on_change("start", self.on_change)
        figure.x_range.on_change("end", self.on_change)
        if self.has_document():
            self.add_periodic_callback()

    def on_change(self, attr, old, new):
        if not self.has_document():
            return
        if not self.has_periodic_callback():
            self.add_periodic_callback()

    def has_document(self):
        return self.figure.document is not None

    def has_periodic_callback(self):
        return self.pcb is not None

    def add_periodic_callback(self):
        self.pcb = self.figure.document.add_periodic_callback(self.redraw, self.interval)

    def remove_periodic_callback(self):
        if self.pcb is None:
            return
        self.figure.document.remove_periodic_callback(self.pcb)
        self.pcb = None

    def redraw(self):
        extent = (
            self.figure.x_range.start,
            self.figure.x_range.end,
            self.figure.y_range.start,
            self.figure.y_range.end
        )
        if self.valid(extent):
            self.draw(extent)
        if self.has_periodic_callback():
            self.remove_periodic_callback()

    def valid(self, extent):
        return all(value is not None for value in extent)

    def draw(self, extent):
        xs, ys = multi_lines(self.feature, extent)
        self.source.data = {
            "xs": xs,
            "ys": ys
        }


def coastlines(extent, scale="50m"):
    """Add cartopy coastline to a figure

    Translates cartopy.feature.COASTLINE object
    into collection of lines

    .. note:: This method assumes the map projection
              is cartopy.crs.PlateCarreee

    :param scale: cartopy scale '110m', '50m' or '10m'
    :param extent: x_start, x_end, y_start, y_end
    """
    feature = cartopy.feature.COASTLINE
    feature.scale = scale
    return multi_lines(feature, extent)


def borders(extent, scale="50m"):
    """Add cartopy borders to a figure

    Translates cartopy.feature.BORDERS feature
    into collection of lines

    .. note:: This method assumes the map projection
              is cartopy.crs.PlateCarreee

    :param scale: cartopy scale '110m', '50m' or '10m'
    """
    feature = cartopy.feature.BORDERS
    feature.scale = scale
    return multi_lines(feature, extent)


def multi_lines(feature, extent):
    xs, ys = [], []
    for geometry in feature.intersecting_geometries(extent):
        for g in geometry:
            x, y = g.xy
            x,y = np.asarray(x), np.asarray(y)
        mask = ~inside(x, y, extent)
        x = np.ma.masked_array(x, mask)
        y = np.ma.masked_array(y, mask)
        xs.append(x)
        ys.append(y)
    return xs, ys


def inside(x, y, extent):
    """Detect points inside extent"""
    x, y = np.asarray(x), np.asarray(y)
    x_start, x_end, y_start, y_end = extent
    return ((x > x_start) &
            (x < x_end) &
            (y > y_start) &
            (y < y_end))
