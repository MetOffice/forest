import cartopy
import numpy as np


def to_180(x):
    y = x.copy()
    y[y > 180.] -= 360.
    return y


def web_mercator(lons, lats):
    return transform(
            lons,
            lats,
            cartopy.crs.PlateCarree(),
            cartopy.crs.Mercator.GOOGLE)


def plate_carree(x, y):
    return transform(
            x,
            y,
            cartopy.crs.Mercator.GOOGLE,
            cartopy.crs.PlateCarree())


def transform(x, y, src_crs, dst_crs):
    x, y = np.asarray(x), np.asarray(y)
    xt, yt, _ = dst_crs.transform_points(src_crs, x.flatten(), y.flatten()).T
    return xt, yt
