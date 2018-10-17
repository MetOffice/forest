import unittest
import numpy as np
import datetime as dt
import bokeh.models
import forest


def date_range(initial, hours, n):
    if not isinstance(hours, dt.timedelta):
        hours = dt.timedelta(hours=hours)
    return [initial + i * hours for i in range(n)]


class TestForecast(unittest.TestCase):
    def setUp(self):
        self.forecast = forest.Forecast.stamp("2018-01-01 12:00", 1)
        self.forecasts = [
            forest.Forecast.stamp("2018-01-01 00:00", 24),
            forest.Forecast.stamp("2018-01-01 00:00", 12),
            forest.Forecast.stamp("2018-01-02 00:00", 0),
            forest.Forecast.stamp("2018-01-02 00:00", 12),
        ]

    def test_valid_time(self):
        result = self.forecast.valid_time
        expect = dt.datetime(2018, 1, 1, 13)
        self.assertEqual(expect, result)

    def test_label(self):
        result = self.forecast.label()
        expect = "2018-01-01 12:00 T+1"
        self.assertEqual(expect, result)

    def test_filter_forecasts_by_valid_time(self):
        valid_time = dt.datetime(2018, 1, 2)
        result = [f for f in self.forecasts
                  if f.valid_time == valid_time]
        expect = [
            forest.Forecast.stamp("2018-01-01 00:00", 24),
            forest.Forecast.stamp("2018-01-02 00:00", 0),
        ]
        self.assertEqual(expect, result)

    def test_filter_forecasts_by_length(self):
        result = [f for f in self.forecasts
                  if f.length == dt.timedelta(hours=12)]
        expect = [
            forest.Forecast.stamp("2018-01-01 00:00", 12),
            forest.Forecast.stamp("2018-01-02 00:00", 12),
        ]
        self.assertEqual(expect, result)

    def test_index_given_list_of_forecasts(self):
        forecast = forest.Forecast.stamp("2018-01-01 00:00", 12)
        result = self.forecasts.index(forecast)
        expect = 1
        self.assertEqual(expect, result)


class TestPeriod(unittest.TestCase):
    def test_period(self):
        period = forest.Period.stamps("2018-01-01 00:00",
                                      "2018-01-01 03:00")
        self.assertEqual(period.start, dt.datetime(2018, 1, 1))
        self.assertEqual(period.end, dt.datetime(2018, 1, 1, 3))

    def test_duration(self):
        period = forest.Period.stamps("2018-01-01 00:00",
                                      "2018-01-01 03:00")
        self.assertEqual(period.duration, dt.timedelta(hours=3))

    def test_equality(self):
        period_1 = forest.Period.stamps("2018-01-01 00:00",
                                        "2018-01-01 03:00")
        period_2 = forest.Period.stamps("2018-01-01 00:00",
                                        "2018-01-01 03:00")
        self.assertEqual(period_1, period_2)

    def test_label(self):
        period = forest.Period.stamps("2018-01-01 00:00",
                                      "2018-01-01 03:00")
        result = period.label("{:%H:%M} - {:%H:%M}")
        expect = "00:00 - 03:00"
        self.assertEqual(expect, result)


class TestNavigate(unittest.TestCase):
    def test_forecasts_given_times(self):
        initial = dt.datetime(2018, 1, 1)
        text = "2018-01-01 00:00"
        hours = dt.timedelta(hours=3)
        n = 2
        times = date_range(initial, hours, n)
        result = forest.navigate.forecasts(times)
        expect = [forest.Forecast.stamp(text, 0),
                  forest.Forecast.stamp(text, 3)]
        self.assertEqual(expect, result)

    def test_paragraph_updated_by_stream(self):
        stream = forest.Stream()
        p = bokeh.models.Paragraph()
        stream.map(forest.navigate.text(p))
        stream.emit("Hello, world!")
        result = p.text
        expect = "Hello, world!"
        self.assertEqual(expect, result)

    def test_emit_stream_closure(self):
        stream = forest.Stream()
        stream.emit = unittest.mock.Mock()
        emit = forest.navigate.emit(stream, +1)
        emit()
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
