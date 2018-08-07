"""Geographical features, coast lines and political boundaries"""
import numpy as np
import cartopy


def add_coastlines(figure, extent):
    xs, ys = [], []
    for x, y in coastlines(scale="50m"):
        x, y = clip_xy(x, y, extent)
        if x.shape[0] == 0:
            continue
        xs.append(x)
        ys.append(y)
    figure.multi_line(xs, ys,
                      color='black',
                      level='overlay')


def add_borders(figure, extent):
    xs, ys = [], []
    for x, y in borders(scale="50m"):
        x, y = clip_xy(x, y, extent)
        if x.shape[0] == 0:
            continue
        xs.append(x)
        ys.append(y)
    figure.multi_line(xs, ys,
                      color='grey',
                      level='overlay')


def coastlines(scale="110m"):
    """Add cartopy coastline to a figure

    Translates cartopy.feature.COASTLINE object
    into collection of lines

    .. note:: This method assumes the map projection
              is cartopy.crs.PlateCarreee

    :param scale: cartopy scale '110m', '50m' or '10m'
    :param extent: x_start, x_end, y_start, y_end
    """
    return feature_lines(cartopy.feature.COASTLINE,
                         scale=scale)


def borders(scale="110m"):
    """Add cartopy borders to a figure

    Translates cartopy.feature.BORDERS feature
    into collection of lines

    .. note:: This method assumes the map projection
              is cartopy.crs.PlateCarreee

    :param scale: cartopy scale '110m', '50m' or '10m'
    """
    return feature_lines(cartopy.feature.BORDERS,
                         scale=scale)


def feature_lines(feature, scale="110m"):
    feature.scale = scale
    for geometry in feature.geometries():
        for g in geometry:
            x, y = g.xy
            yield np.asarray(x), np.asarray(y)


def clip_xy(x, y, extent):
    """Clip coastline to be inside figure extent"""
    x, y = np.asarray(x), np.asarray(y)
    x_start, x_end, y_start, y_end = extent
    pts = np.where((x > x_start) &
                   (x < x_end) &
                   (y > y_start) &
                   (y < y_end))
    return x[pts], y[pts]


def cutout_xy(x, y, extent):
    """Cutout x, y inside a box"""
    x, y = np.asarray(x), np.asarray(y)
    x_start, x_end, y_start, y_end = extent
    pts = np.where((x <= x_start) |
                   (x >= x_end) |
                   (y <= y_start) |
                   (y >= y_end))
    return x[pts], y[pts]


def box_split(x, y, extent):
    """Split a polygonal chain into segments inside/outside box"""
    pts = np.where(np.diff(in_box(x, y, extent)))[0]
    if len(pts) == 0:
        return [[x, y]]
    return zip(np.split(x, pts + 1), np.split(y, pts + 1))


def in_box(x, y, extent):
    x, y = np.asarray(x), np.asarray(y)
    x_start, x_end, y_start, y_end = extent
    return ((x <= x_start) |
            (x >= x_end) |
            (y <= y_start) |
            (y >= y_end))
