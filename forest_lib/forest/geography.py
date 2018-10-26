"""Geographical features, coast lines and political boundaries"""
import numpy as np
import cartopy

__all__ = [
    "bounding_square",
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


def coastlines(extent, scale="50m"):
    """Add cartopy coastline to a figure

    Translates cartopy.feature.COASTLINE object
    into collection of lines

    .. note:: This method assumes the map projection
              is cartopy.crs.PlateCarreee

    :param scale: cartopy scale '110m', '50m' or '10m'
    :param extent: x_start, x_end, y_start, y_end
    """
    xs, ys = [], []
    for x, y in feature_lines(cartopy.feature.COASTLINE,
                              scale=scale):
        x, y = clip_xy(x, y, extent)
        if x.shape[0] == 0:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def borders(extent, scale="50m"):
    """Add cartopy borders to a figure

    Translates cartopy.feature.BORDERS feature
    into collection of lines

    .. note:: This method assumes the map projection
              is cartopy.crs.PlateCarreee

    :param scale: cartopy scale '110m', '50m' or '10m'
    """
    xs, ys = [], []
    for x, y in feature_lines(cartopy.feature.BORDERS,
                              scale=scale):
        x, y = clip_xy(x, y, extent)
        if x.shape[0] == 0:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


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
