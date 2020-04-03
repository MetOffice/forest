"""
Geographic utilities module
---------------------------

Module to handle projection and sampling iof points for imaging.

.. autofunction:: stretch_image

.. autofunction:: web_mercator

.. autofunction:: plate_carree

"""
try:
    import cartopy
except ImportError:
    # ReadTheDocs unable to pip install cartopy
    pass

import numpy as np

import scipy.interpolate
import scipy.ndimage

try:
    import datashader
    import xarray
except ModuleNotFoundError:
    datashader = None

def stretch_image(lons, lats, values,
                  plot_height=None,
                  plot_width=None):
    """
    Do the mapping from image data to the format required by bokeh
    for plotting.

    :param lons: Numpy array with latitude values for the image data.
    :param lats: Numpy array with longitude values for the image data.
    :param values: Numpy array of image data, with dimensions matching the
                   size of latitude and longitude arrays.
    :return: A dictionary that can be used with the bokeh image glyph.
    """
    if (lons.ndim == 1):
        gx, _ = web_mercator(
            lons,
            np.zeros(len(lons), dtype="d"))
        _, gy = web_mercator(
            np.zeros(len(lats), dtype="d"),
            lats)
    elif (lons.ndim == 2) and (lats.ndim == 2):
        gx, gy = web_mercator(lons, lats)
        gx = gx.reshape(lons.shape)
        gx = np.ma.masked_invalid(gx)
        gy = gy.reshape(lats.shape)
        gy = np.ma.masked_invalid(gy)
    else:
        raise Exception("Either 1D or 2D lons/lats")

    if datashader:
        x_range = (gx.min(), gx.max())
        y_range = (gy.min(), gy.max())
        image = datashader_stretch(values, gx, gy, x_range, y_range,
                                   plot_height=plot_height,
                                   plot_width=plot_width)
    else:
        # TODO: Deprecate this method
        image = custom_stretch(values, gx, gy)

    # Image location
    x = gx.min()
    y = gy.min()
    if gx.ndim == 1:
        # 1D image extent
        dw = gx[-1] - gx[0]
        dh = gy[-1] - gy[0]
    else:
        # 2D image extent
        dw = gx.max() - gx.min()
        dh = gy.max() - gy.min()
    return {
        "x": [x],
        "y": [y],
        "dw": [dw],
        "dh": [dh],
        "image": [image]
    }


def datashader_stretch(values, gx, gy, x_range, y_range,
                       plot_height=None,
                       plot_width=None):
    """
    Use datashader to sample the data mesh in on a regular grid for use in
    image display.

    :param values: A numpy array of image data
    :param gx: The array of coordinates in projection space.
    :param gy: The array of coordinates in projection space.
    :param x_range: The range of the mesh in projection space.
    :param y_range: The range of the mesh in projection space.
    :return: An xarray of image data representing pixels.
    """
    if plot_height is None:
        plot_height = values.shape[0]
    if plot_width is None:
        plot_width = values.shape[1]
    canvas = datashader.Canvas(plot_height=plot_height,
                               plot_width=plot_width,
                               x_range=x_range,
                               y_range=y_range)
    if gx.ndim == 1:
        # 1D Quadmesh
        xarr = xarray.DataArray(values, coords=[('y', gy), ('x', gx)], name='Z')
        image = canvas.quadmesh(xarr)
    else:
        # 2D Quadmesh
        xarr = xarray.DataArray(values,
                                dims=['Y', 'X'],
                                coords={
                                    'Qx': (['Y', 'X'], gx),
                                    'Qy': (['Y', 'X'], gy)
                                },
                                name='Z')
        image = canvas.quadmesh(xarr, x='Qx', y='Qy')
    return np.ma.masked_array(image.values,
                          mask=np.isnan(
                              image.values))

def custom_stretch(values, gx, gy):
    if np.ma.is_masked(values):
        mask = values.mask
    else:
        mask = None
    image = stretch_y(gy)(values)
    if mask is not None:
        image_mask = stretch_y(gy)(mask)
        image = np.ma.masked_invalid(
            np.ma.masked_array(image, mask=image_mask))
    return image


def stretch_y(uneven_y):
    """Mercator projection stretches longitude spacing

    To remedy this effect an even-spaced resampling is performed
    in the projected space to make the pixels and grid line up

    .. note:: This approach assumes the grid is evenly spaced
              in longitude/latitude space prior to projection
    """
    if isinstance(uneven_y, list):
        uneven_y = np.asarray(uneven_y, dtype=np.float)
    even_y = np.linspace(
        uneven_y.min(), uneven_y.max(), len(uneven_y),
        dtype=np.float)
    index = np.arange(len(uneven_y), dtype=np.float)
    index_function = scipy.interpolate.interp1d(uneven_y, index)
    index_fractions = index_function(even_y)

    def wrapped(values, axis=0):
        if isinstance(values, list):
            values = np.asarray(values, dtype=np.float)
        assert values.ndim == 2, "Can only stretch 2D arrays"
        msg = "{} != {} do not match".format(values.shape[axis], len(uneven_y))
        assert values.shape[axis] == len(uneven_y), msg
        if axis == 0:
            i = index_fractions
            j = np.arange(values.shape[1], dtype=np.float)
        elif axis == 1:
            i = np.arange(values.shape[0], dtype=np.float)
            j = index_fractions
        else:
            raise Exception("Can only handle axis 0 or 1")
        return scipy.ndimage.map_coordinates(
            values,
            np.meshgrid(i, j, indexing="ij"),
            order=1)
    return wrapped


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
