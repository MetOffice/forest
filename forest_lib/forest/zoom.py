"""
The simplest zoom feature can be implemented by rendering
a high resolution image overlay on top of a coarse resolution
image. The effect for the user is an initally coarse image
is clarified as a high resolution image after a slight delay. For
the developer the high resolution patch can be maintained
independently from the coarse resolution image, thus reducing
the burden of managing images

Careful management of the the high resolution patch is important
for memory/performance reasons. If memory/performance were not an
issue one could render a high resolution full domain image and
be done

.. note:: ``forest.ForestPlot`` routinely replaces
          ``bokeh_image.data_source`` with a length 1 ColumnDataSource,
          the zoom tool should detect when this is the case and add
          an overlay regardless

"""
import datetime as dt
from functools import wraps


__all__ = ["Zoom"]


class throttle(object):
    """Decorator to limit frequency of method calls"""
    def __init__(self, milliseconds=0):
        self.period = dt.timedelta(milliseconds=milliseconds)
        self.last_call = dt.datetime.min

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            now = dt.datetime.now()
            duration = now - self.last_call
            if duration > self.period:
                self.last_call = now
                return fn(*args, **kwargs)
        return wrapper


class Zoom(object):
    """General mechanism to connect bokeh figure to render_method

    Zoom mechanism connects x_range and y_range on_change
    callbacks to a render_method with the optional ability
    to throttle the frequency of callbacks

    .. note:: render_method must have call signature
              ``method(x_start, x_end, y_start, y_end)``
    """
    def __init__(self, render_method, throttle_milliseconds=None):
        self.render_method = render_method
        if throttle_milliseconds is not None:
            milliseconds = throttle_milliseconds
            self.render = throttle(milliseconds=milliseconds)(self.render)

    def add_figure(self, figure):
        self.figure = figure
        self.figure.x_range.on_change("start", self.on_change)
        self.figure.x_range.on_change("end", self.on_change)
        self.figure.y_range.on_change("start", self.on_change)
        self.figure.y_range.on_change("end", self.on_change)

    def on_change(self, attr, old, new):
        """Bokeh callback interface"""
        self.render(self.figure.x_range.start, self.figure.x_range.end,
                    self.figure.y_range.start, self.figure.y_range.end)

    def render(self, x_start, x_end, y_start, y_end):
        """Method to add high-resolution imagery to figure"""
        self.render_method(x_start, x_end, y_start, y_end)
