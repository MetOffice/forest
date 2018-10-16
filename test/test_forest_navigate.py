import unittest
import numpy as np
import datetime as dt
import bokeh.models
import forest


class TestNavigate(unittest.TestCase):
    def test_forecast_view(self):
        widget = forest.navigate.forecast_view()
        self.assertIsInstance(widget, bokeh.layouts.Row)

    def test_on_click_stream_closure(self):
        stream = forest.Stream()
        stream.emit = unittest.mock.Mock()
        on_click = forest.navigate.on_click(stream, +1)
        on_click()
        stream.emit.assert_called_once_with(+1)

    def test_forecast_lengths(self):
        times = [dt.datetime(2018, 1, 1),
                 dt.datetime(2018, 1, 1, 3)]
        result = forest.navigate.forecast_lengths(times)
        expect = [dt.timedelta(hours=0),
                  dt.timedelta(hours=3)]
        np.testing.assert_array_equal(expect, result)

    def test_initial_time(self):
        times = [dt.datetime(2018, 1, 1),
                 dt.datetime(2018, 1, 1, 3)]
        result = forest.navigate.initial_time(times)
        expect = dt.datetime(2018, 1, 1)
        np.testing.assert_array_equal(expect, result)

    def test_forecast_label(self):
        time, hour = dt.datetime(2018, 1, 1), 12
        result = forest.navigate.forecast_label(time, hour)
        expect = "2018-01-01 00:00 T+12"
        self.assertEqual(expect, result)
