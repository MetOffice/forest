"""Tools to compare RGBA images on a pixel by pixel basis

The tools in this module are designed to work with
:meth:`bokeh.plotting.figure.Figure.image_rgba` GlyphRenderers and
ColumnDataSources that drive those images

A :class:`.Slider` is available that makes it easy to compare two
images relative to a mouse pointer. The slider splits images
such that the left portion of one image and
the right portion of the other image are visible either side of the mouse
position.

>>> slider = forest.image.Slider(left_images, right_images)
>>> slider.add_figure(figure)

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

The :class:`Zoom` class represents a mechanism to generate high-resolution
imagery as the x/y ranges change. The particular implementation available
here merely displays portions of a higher resolution array overlayed on a
coarsified full extent image

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


class Zoom(object):
    """Tool to generate high-resolution overlays as x/y ranges change"""


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

    def on_change(self, attr, old, new):
        """Interface to bokeh.widgets on_change callback

        Hides/shows appropriate images related to button group

        .. note:: ``attr`` and ``old`` parameters are ignored by
                  this callback

        :param new: integer 0 indicating display left image, 1 indicates
                    right image to be displayed
        """
        if new == 0:
            self.show(self.left_images)
            self.hide(self.right_images)
        else:
            self.show(self.right_images)
            self.hide(self.left_images)

    def show(self, images):
        """Show an image"""
        if "_alpha" not in images.data:
            images.data["_alpha"] = []

    def hide(self, images):
        """Hide an image"""
        pass


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
        self.left_images = left_images
        self.left_extra = bokeh.models.ColumnDataSource({
            "alpha": self.get_alpha(self.left_images),
            "shape": self.get_shapes(self.left_images)
        })
        self.right_images = right_images
        self.right_extra = bokeh.models.ColumnDataSource({
            "alpha": self.get_alpha(self.right_images),
            "shape": self.get_shapes(self.right_images)
        })
        self.span = bokeh.models.Span(location=0,
                                      dimension='height',
                                      line_color='black',
                                      line_width=1)
        shared = bokeh.models.ColumnDataSource({
            "first_time": [False],
            "previous_mouse_x": [0]
        })
        self.mousemove = bokeh.models.CustomJS(args=dict(
            span=self.span,
            left_images=self.left_images,
            left_extra=self.left_extra,
            right_images=self.right_images,
            right_extra=self.right_extra,
            shared=shared
        ), code=JS_CODE)

    def get_alpha(self, bokeh_obj):
        """Helper to copy alpha values from GlyphRenderer or ColumnDataSource"""
        images = self.get_images(bokeh_obj)
        return [np.copy(rgba[:, :, -1]) for rgba in images]

    def get_shapes(self, bokeh_obj):
        """Helper to shapes from GlyphRenderer or ColumnDataSource"""
        images = self.get_images(bokeh_obj)
        return [rgba.shape for rgba in images]

    @staticmethod
    def get_images(bokeh_obj):
        if hasattr(bokeh_obj, "data_source"):
            renderer = bokeh_obj
            if isinstance(renderer.glyph.image, basestring):
                return renderer.data_source[renderer.glyph.image]
            else:
                return renderer.glyph.image
        column_data_source = bokeh_obj
        return column_data_source.data["image"]

    def add_figure(self, figure):
        """Attach various callbacks to a particular figure"""
        hover_tool = bokeh.models.HoverTool(callback=self.mousemove)
        figure.add_tools(hover_tool)
        figure.renderers.append(self.span)
