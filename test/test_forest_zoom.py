import unittest
import unittest.mock
import bokeh.models
import bokeh.plotting
import forest.zoom


class TestOverlay(unittest.TestCase):
    """Maintain high-resolution patch"""
    def test_can_be_constructed(self):
        forest.zoom.Overlay()


class FakeCallback(object):
    """Bokeh callback signature but logs calls"""
    def __init__(self):
        self.called = False
    def __call__(self, attr, old, new):
        self.called = True
        self.call_signature = attr, old, new
    def assert_called_once_with(self, attr, old, new):
        assert self.called, "{} not called".format(self.__repr__())
        actual_call = self.call_signature
        expected_call = (attr, old, new)
        message = "{} != {}".format(actual_call, expected_call)
        assert actual_call == expected_call, message


class TestForestZoom(unittest.TestCase):
    def setUp(self):
        self.figure = bokeh.plotting.figure()
        self.render_method = None
        self.zoom = forest.zoom.Zoom(self.render_method)

    def test_add_figure_connects_x_range_start_to_on_change(self):
        self.check_add_figure_connects("x_range", "start", None, 100)

    def test_add_figure_connects_x_range_end_to_on_change(self):
        self.check_add_figure_connects("x_range", "end", None, 100)

    def test_add_figure_connects_y_range_start_to_on_change(self):
        self.check_add_figure_connects("y_range", "start", None, 100)

    def test_add_figure_connects_y_range_end_to_on_change(self):
        self.check_add_figure_connects("y_range", "end", None, 100)

    def check_add_figure_connects(self, dim_range, attr, old, new):
        self.zoom.on_change = FakeCallback()
        self.zoom.add_figure(self.figure)
        getattr(self.figure, dim_range).trigger(attr, old, new)
        self.zoom.on_change.assert_called_once_with(attr, old, new)

    def test_on_change_calls_render_with_axis_limits(self):
        attr, old, new = "start", None, 100 # irrelevant to method
        x_start, x_end = 0, 1
        y_start, y_end = 10, 20
        self.figure.x_range.start = x_start
        self.figure.x_range.end = x_end
        self.figure.y_range.start = y_start
        self.figure.y_range.end = y_end
        self.zoom.add_figure(self.figure)
        self.zoom.render = unittest.mock.Mock()
        self.zoom.on_change(attr, old, new)
        self.zoom.render.assert_called_once_with(x_start, x_end,
                                                 y_start, y_end)


class TestThrottle(unittest.TestCase):
    """Decorator to throttle function calls"""
    def test_throttle_calls_method_with_args_kwargs(self):
        args, kwargs = ("hello",), {"name": "world"}
        method = unittest.mock.Mock()
        throttled_method = forest.zoom.throttle()(method)
        throttled_method(*args, **kwargs)
        method.assert_called_once_with(*args, **kwargs)

    def test_throttle_called_twice_rapidly(self):
        args, kwargs = ("hello",), {"name": "world"}
        method = unittest.mock.Mock()
        @forest.zoom.throttle(milliseconds=100)
        def throttled_method(*args, **kwargs):
            return method(*args, **kwargs)
        throttled_method(*args, **kwargs)
        throttled_method(*args, **kwargs)
        method.assert_called_once_with(*args, **kwargs)
