"""Slider tool"""
import numpy as np
import bokeh.models

with open("slide.js", "r") as stream:
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
