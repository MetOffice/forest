"""Geographical features, coast lines and political boundaries"""
import bokeh.models
import numpy as np
import cartopy

__all__ = [
    "bounding_square",
    "coastlines"
]


def export(obj):
    if obj.__name__ not in __all__:
        __all__.append(obj.__name__)
    return obj


@export
def web_mercator(lons, lats):
    """Web Mercator projection standardised by Google"""
    return transform(cartopy.crs.PlateCarree(), cartopy.crs.Mercator.GOOGLE, lons, lats)


@export
def transform(source_crs, target_crs, lons, lats):
    lons = np.ma.asarray(lons)
    lats = np.ma.asarray(lats)
    xyz = target_crs.transform_points(
        source_crs,
        lons,
        lats
    )
    x, y, _ = xyz.T
    x = np.ma.masked_array(x, mask=lons.mask)
    y = np.ma.masked_array(y, mask=lats.mask)
    if lons.ndim == 0:
        return x[0], y[0]
    return x, y


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


@export
def add_coastlines(figure, scale="50m", **kwargs):
    feature = cartopy.feature.COASTLINE
    feature.scale = scale
    return add_feature(figure, feature, **kwargs)


@export
def add_borders(figure, scale="50m", **kwargs):
    feature = cartopy.feature.BORDERS
    feature.scale = scale
    return add_feature(figure, feature, **kwargs)


def add_feature(figure, feature, color="black", projection=None):
    """Add cartopy feature to bokeh figure

    :param projection: target coordinate system for plot
    """
    source = bokeh.models.ColumnDataSource({
        "xs": [],
        "ys": []
    })
    figure.multi_line(xs="xs", ys="ys",
                      source=source,
                      color=color)
    def draw(x_start, x_end, y_start, y_end):
        xy_extent = x_start, x_end, y_start, y_end
        if projection is None:
            xys = multi_lines(feature, xy_extent)
        else:
            # Map between coordinate systems
            plate_carree = cartopy.crs.PlateCarree()
            x = [x_start, x_end]
            y = [y_start, y_end]
            lons, lats = transform(projection, plate_carree, x, y)
            lonlat_extent = (lons[0], lons[1], lats[0], lats[1])
            xys = (transform(plate_carree, projection, lon, lat) for lon, lat
                    in multi_lines(feature, lonlat_extent))
        xs, ys = join(xys)
        source.data = {
            "xs": xs,
            "ys": ys
        }
    return PeriodicArtist(figure, draw)


class PeriodicArtist(object):
    def __init__(self, figure, draw, interval=500):
        self.pcb = None
        self.figure = figure
        self.draw = draw
        self.interval = interval
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
        if self.valid(*extent):
            self.draw(*extent)
        if self.has_periodic_callback():
            self.remove_periodic_callback()

    def valid(self, *values):
        return all(value is not None for value in values)


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
    return join(multi_lines(feature, extent))


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
    return join(multi_lines(feature, extent))


def multi_lines(feature, extent):
    for geometry in feature.intersecting_geometries(extent):
        for g in geometry:
            x, y = g.xy
            x,y = np.asarray(x), np.asarray(y)
        mask = ~inside(x, y, extent)
        x = np.ma.masked_array(x, mask)
        y = np.ma.masked_array(y, mask)
        yield x, y


def join(xys):
    xs, ys = [], []
    for x, y in xys:
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
