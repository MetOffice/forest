"""

Image comparison tools
======================

Tools to compare RGBA images on a pixel by pixel basis

References to the :class:`bokeh.models.sources.ColumnDataSource` sources
related to the left/right images should be provided by the user. This
is typically the responsibility of some glue code, either a ``main.py``
or some other calling code

.. note:: The algorithms and data structures expose methods
          that can be attached to bokeh events. They are
          unaware of the calling code architecture and simply
          edit appropriate RGBA values

Slider
------

A :class:`.Slider` is available that makes it easy to compare two
images relative to a mouse pointer. The slider splits images
such that the left portion of one image and
the right portion of the other image are visible either side of the mouse
position.

>>> slider = forest.image.Slider(left_images, right_images)
>>> slider.add_figure(figure)

Application programming interface (API)
---------------------------------------

The following classes have been made available to users
of Forest for custom visualisations
"""
import os
import sys
import numpy as np
import bokeh.models
import bokeh.events


# CustomJS callback code
JS_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "image.js")
with open(JS_FILE, "r") as stream:
    JS_CODE = stream.read()


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
        self.span = bokeh.models.Span(location=0,
                                      dimension='height',
                                      line_color='black',
                                      line_width=1)
        self.shared = bokeh.models.ColumnDataSource({
            "use_previous_mouse_x": [False],
            "previous_mouse_x": [0]
        })
        self.mousemove = bokeh.models.CustomJS(args=dict(
            span=self.span,
            left_images=self.left_images,
            right_images=self.right_images,
            shared=self.shared
        ), code=JS_CODE)

    @property
    def left_images(self):
        return self.images["left"]

    @property
    def right_images(self):
        return self.images["right"]

    def add_figure(self, figure):
        """Attach various callbacks to a particular figure"""
        self.hover_tool = bokeh.models.HoverTool(callback=self.mousemove)
        figure.add_tools(self.hover_tool)
        figure.renderers.append(self.span)

        # Add MouseEnter callback to reset slider
        callback = bokeh.models.CustomJS(args=dict(shared=self.shared), code="""
            shared.data.use_previous_mouse_x[0] = false;
        """)
        figure.js_on_event(bokeh.events.MouseEnter, callback)


def get_shapes(source):
    """Read image shapes from ColumnDataSource"""
    images = source.data["image"]
    return [image.shape for image in images]


def get_alpha(bokeh_obj):
    """Helper to copy alpha values from GlyphRenderer or ColumnDataSource"""
    images = get_images(bokeh_obj)
    return [np.copy(rgba[..., -1]) for rgba in images]


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
