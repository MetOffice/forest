"""

Image comparison tools
======================

Tools to compare RGBA images on a pixel by pixel basis

The tools in this module are designed to work with
:meth:`bokeh.plotting.figure.Figure.image_rgba` GlyphRenderers and
ColumnDataSources that drive those images

Slider
------

A :class:`.Slider` is available that makes it easy to compare two
images relative to a mouse pointer. The slider splits images
such that the left portion of one image and
the right portion of the other image are visible either side of the mouse
position.

>>> slider = forest.image.Slider(left_images, right_images)
>>> slider.add_figure(figure)

Toggle
------

A :class:`.Toggle` is also available to switch between images.  Instead of
displaying portions of two images side by side, like the slider, the toggle
swaps out images to quickly see the differences between each image

>>> toggle = forest.image.Toggle(left_images, right_images)
>>> buttons = bokeh.models.widgets.RadioButtonGroup(
...     labels=["left", "right"],
...     active=0
... )
>>> buttons.on_change("active", toggle.on_change)

Importantly, a :class:`Toggle` has no knowledge of bokeh widgets
or layouts, it simply responds to ``on_change`` events by editing
the appropriate alpha values of the associated images

Application programming interface (API)
=======================================

The following classes have been made available to users
of Forest for custom visualisations
"""
import os
import sys
import numpy as np
import bokeh.models


# CustomJS callback code
JS_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "image.js")
with open(JS_FILE, "r") as stream:
    JS_CODE = stream.read()


class Toggle(object):
    """Controls alpha values of bokeh RGBA ColumnDataSources

    Similar design to :class:`.Slider` but with wholesale replacement
    of image alpha values

    :param left_images: ColumnDataSource or GlyphRenderer used to
                        define RGBA images when toggle is set to left
    :param right_images: ColumnDataSource or GlyphRenderer used to
                         define RGBA images when toggle is set to right
    """
    def __init__(self, left_images, right_images):
        self.left_images = left_images
        self.right_images = right_images

    def show_left(self):
        """Show left image and hide right image"""
        self.show(self.left_images)
        self.hide(self.right_images)

    def show_right(self):
        """Show right image and hide left image"""
        self.show(self.right_images)
        self.hide(self.left_images)

    def show(self, source):
        """Show an image

        .. note:: Caches existing alpha values if not already cached
                  and exits since there is no alpha information to
                  change
        .. note:: Updates image alpha values in-place
        """
        if "_alpha" not in source.data:
            cache_alpha(source)
            return
        # Restore alpha from cache
        images = source.data["image"]
        alphas = source.data["_alpha"]
        for image, alpha in zip(images, alphas):
            image[..., -1] = alpha
        source.data["image"] = images

    def hide(self, source):
        """Hide an image

        Set alpha values in RGBA arrays to zero, while keeping a
        copy of the original values for restoration purposes

        .. note:: Caches existing alpha values if not already cached
        .. note:: Updates image alpha values in-place
        """
        if "_alpha" not in source.data:
            cache_alpha(source)
        # Set alpha to zero
        images = source.data["image"]
        for image in images:
            image[..., -1] = 0
        source.data["image"] = images


class Slider(object):
    """Controls alpha values of bokeh RGBA ColumnDataSources

    Understands bokeh architecture, either a GlyphRenderer
    or a ColumnDataSource may be passed in for either
    left_images or right_images arguments

    The various properties needed to set alpha values relative
    to the mouse position can be found by inspecting
    either data structure

    .. note:: If a ColumnDataSource is passed then the following
              keys are used to find the image properties
              'image', 'x', 'y', 'dw', 'dh'

    :param left_images: ColumnDataSource or GlyphRenderer used to
                        define RGBA images to the left of the cursor
    :param right_images: ColumnDataSource or GlyphRenderer used to
                         define RGBA images to the right of the cursor
    """
    def __init__(self, left_images, right_images):
        self.images = {
            "left": left_images,
            "right": right_images
        }
        self.shapes = {
            "left": image_shapes(self.left_images),
            "right": image_shapes(self.right_images)
        }
        for side in ["left", "right"]:
            if "_alpha" not in self.images[side].data:
                cache_alpha(self.images[side])

            self.shapes[side] = image_shapes(self.images[side])
            if "_shape" not in self.images[side].data:
                self.images[side].data["_shape"] = self.shapes[side]

        self.span = bokeh.models.Span(location=0,
                                      dimension='height',
                                      line_color='black',
                                      line_width=1)
        shared = bokeh.models.ColumnDataSource({
            "use_previous_mouse_x": [True],
            "previous_mouse_x": [0]
        })
        self.mousemove = bokeh.models.CustomJS(args=dict(
            span=self.span,
            left_images=self.left_images,
            right_images=self.right_images,
            shared=shared
        ), code=JS_CODE)

        # Listen to server-side image changes
        self.left_images.on_change("data", self.on_change)
        self.right_images.on_change("data", self.on_change)

    @property
    def left_images(self):
        return self.images["left"]

    @property
    def right_images(self):
        return self.images["right"]

    def on_change(self, attr, old, new):
        """Listen for bokeh server-side image array changes"""
        for side in ["left", "right"]:
            # Listen for shape changes
            shapes = image_shapes(self.images[side])
            if tuple(self.shapes[side]) != tuple(shapes):
                # Note: order important to prevent infinite recursion
                self.shapes[side] = shapes
                self.images[side].data["_shape"] = shapes

        # Pass latest image shapes to client-side
        self.mousemove.args["left_images"] = self.left_images
        self.mousemove.args["right_images"] = self.right_images

    def add_figure(self, figure):
        """Attach various callbacks to a particular figure"""
        self.hover_tool = bokeh.models.HoverTool(callback=self.mousemove)
        figure.add_tools(self.hover_tool)
        figure.renderers.append(self.span)


def cache_alpha(source):
    """Pre-process image to cache alpha values"""
    images = source.data["image"]
    source.data["_alpha"] = [np.copy(image[..., -1]) for image in images]


def image_shapes(source):
    """Read image shapes from ColumnDataSource"""
    images = source.data["image"]
    return [image.shape for image in images]


def get_alpha(bokeh_obj):
    """Helper to copy alpha values from GlyphRenderer or ColumnDataSource"""
    images = get_images(bokeh_obj)
    return [np.copy(rgba[:, :, -1]) for rgba in images]


def get_images(bokeh_obj):
    """Helper to get image data from ColumnDataSource or GlyphRenderer"""
    if hasattr(bokeh_obj, "data_source"):
        renderer = bokeh_obj
        return renderer.data_source.data[renderer.glyph.image]
    column_data_source = bokeh_obj
    return column_data_source.data["image"]


def alpha_from_rgba(rgba):
    """Helper method to view alpha values from an RGBA array

    The alpha values are assumed to be the last entry of the
    last dimension

    .. note:: This method does not handle RGBA data stored as
              1D arrays

    :param rgba: Array of RGBA values
    :returns: View on alpha values
    """
    return rgba[..., -1]
