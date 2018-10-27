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
