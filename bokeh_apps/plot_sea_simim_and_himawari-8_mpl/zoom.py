"""
Zooming and enhancing images in bokeh can be done by rendering
a coarse image for the zoomed out extent and a collection
of high resolution patches as the axis extents get shorter

Careful management of both the number and reuse of the high
resolution patches is important for memory/performance reasons.
If memory were not an issue one could just render the highest
resolution image across the full domain

A first in first out (FIFO) cache can be used to manage
the high resolution patches. The most stale patch can
be replaced by the most current patch, thus keeping the
number of patches constant.

If a zoom view is completely inside an existing patch, then that
patch is good enough for the current view and no extra work
or memory is needed
"""
import bokeh.models
import scipy.ndimage
import numpy as np


class RGBAZoom(object):
    """Coarse/High resolution zoom tool for RGBA images

    .. note:: This is where all the clever FIFO and
              zooming happens

    :param global_rgba: full sized RGBA array to be sub-sampled
                        and sub-viewed
    """
    def __init__(self, global_rgba):
        self.global_rgba = global_rgba
        self.pixels = bokeh.models.ColumnDataSource({
            "x": [0, 0],
            "y": [0, 0]
        })

        # Global NxM pixels (independent of downscaling)
        self.dw = self.global_rgba.shape[1]
        self.dh = self.global_rgba.shape[0]

        # ColumnDataSource to hold high-resolution patches
        self.high_res_source = bokeh.models.ColumnDataSource({
                "image": [],
                "x": [],
                "y": [],
                "dw": [],
                "dh": []
        })

        # Can be constructed without a figure instance
        self.figure = None

    def add_figure(self, figure):
        """Helper method to make RGBAZoom easier to use"""
        figure.x_range.on_change("start", self.zoom_x)
        figure.x_range.on_change("end", self.zoom_x)
        figure.y_range.on_change("start", self.zoom_y)
        figure.y_range.on_change("end", self.zoom_y)
        figure.image_rgba(image="image",
                          x="x",
                          y="y",
                          dw="dw",
                          dh="dh",
                          source=self.high_res_source)
        self.figure = figure

    def zoom_x(self, attr, old, new):
        return self._zoom("x", attr, old, new)

    def zoom_y(self, attr, old, new):
        return self._zoom("y", attr, old, new)

    def _zoom(self, dimension, attr, old, new):
        """General purpose image zoom"""
        # Pixel bounding box for current view
        if attr == "start":
            i = 0
        else:
            i = 1
        pixel = int(new)
        if dimension == "x":
            n = self.dw
        elif dimension == "y":
            n = self.dh
        if pixel > n:
            pixel = n
        if pixel < 0:
            pixel = 0
        self.pixels.data[dimension][i] = pixel

        # Add high resolution imagery
        x = self.pixels.data["x"]
        y = self.pixels.data["y"]
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        if (dx * dy) == 0:
            print("nothing to display")
            return
        if (dx * dy) < 10**6:
            print("plotting high resolution image")
            high_res_image = self.global_rgba[y[0]:y[1], x[0]:x[1]]
            self.high_res_source.data = {
                "image": [high_res_image],
                "x": [x[0]],
                "y": [y[0]],
                "dw": [dx],
                "dh": [dy]
            }
        print("shape:", dx, "x", dy, "pixels:", dx * dy)


def sub_sample(rgba, fraction):
    # Sub-sample imagery since 2000*2000 is too large to work with
    sub_r = scipy.ndimage.zoom(rgba[:, :, 0], fraction)
    sub_g = scipy.ndimage.zoom(rgba[:, :, 1], fraction)
    sub_b = scipy.ndimage.zoom(rgba[:, :, 2], fraction)
    sub_a = scipy.ndimage.zoom(rgba[:, :, 3], fraction)
    result = np.empty((sub_r.shape[0], sub_r.shape[1], 4),
                       dtype=np.uint8)
    result[:, :, 0] = sub_r
    result[:, :, 1] = sub_g
    result[:, :, 2] = sub_b
    result[:, :, 3] = sub_a
    return result


def to_rgba(rgb):
    """Convert RGB to RGBA with alpha set to 1"""
    ni, nj = rgb.shape[0], rgb.shape[1]
    image = np.empty((ni, nj), dtype=np.uint32)
    view = image.view(dtype=np.uint8).reshape((ni, nj, 4))
    view[:, :, 0] = rgb[:, :, 0]
    view[:, :, 1] = rgb[:, :, 1]
    view[:, :, 2] = rgb[:, :, 2]
    view[:, :, 3] = 255
    return view


def is_inside(box_1, box_2):
    """See if box_1 is inside box_2

    This is useful for finding out whether a zoom is looking
    at an internal part of a high resolution patch
    """
    x1, y1, dw1, dh1 = box_1
    x2, y2, dw2, dh2 = box_2
    def between(x, x0, dx):
        return (x0 <= x) & (x <= (x0 + dx))
    return (between(x1, x2, dw2) &
            between(x1 + dw1, x2, dw2) &
            between(y1, y2, dh2) &
            between(y1 + dh1, y2, dh2))


def boxes_overlap(box_1, box_2):
    """Decide if two boxes overlap

    Two boxes overlap if their lower left corner
    is contained in the lowest left-most box of
    the two

    :param box_1: tuple of x, y, dw, dh representing a box
    :param box_2: tuple of x, y, dw, dh representing a box
    :returns: True if boxes overlap
    """
    x1, y1, dw1, dh1 = box_1
    x2, y2, dw2, dh2 = box_2
    if x1 < x2:
        dw = dw1
    else:
        dw = dw2
    if y1 < y2:
        dh = dh1
    else:
        dh = dh2
    return (abs(x1 - x2) < dw) & (abs(y1 - y2) < dh)
