"""Slider tool"""
import numpy as np
import bokeh.models

JS_CODE = """
    /**
     *  Full JS slider implementation
     *     - Hide/show left images
     *     - Hide/show right images
     *     - Move vertical line
     */
    console.log('mouse move');
"""


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
        self.left_alpha = bokeh.models.ColumnDataSource({
            "alpha": self.get_alpha(self.left_images)
        })
        self.right_images = right_images
        self.right_alpha = bokeh.models.ColumnDataSource({
            "alpha": self.get_alpha(self.right_images)
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
            left_alpha=self.left_alpha,
            right_images=self.right_images,
            right_alpha=self.right_alpha,
            shared=shared
        ), code=JS_CODE)

    @staticmethod
    def get_alpha(bokeh_obj):
        """Helper to copy alpha values from RGBA bokeh GlyphRenderer or ColumnDataSource"""
        if hasattr(bokeh_obj, "data_source"):
            renderer = bokeh_obj
            if isinstance(renderer.glyph.image, basestring):
                images = renderer.data_source[renderer.glyph.image]
            else:
                images = renderer.glyph.image
        else:
            column_data_source = bokeh_obj
            images = column_data_source.data["image"]
        return [np.copy(rgba[:, :, -1]) for rgba in images]

    def add_figure(self, figure):
        """Attach various callbacks to a particular figure"""
        figure.js_on_event("mousemove", self.mousemove)
