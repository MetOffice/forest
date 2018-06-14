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
import numpy as np


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
