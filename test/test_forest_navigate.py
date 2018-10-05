import unittest
import datetime as dt
import numpy as np


class TimeControl(object):
    """Forecast/time navigation controller

    Navigates through forecast/time space, by
    either keeping time fixed, forecast length fixed
    or model run fixed
    """
    def __init__(self, forecast_times):
        self.forecast_times = np.asarray(forecast_times, dtype=object)
        self.run_index = 0
        self.forecast_index = 0

    @property
    def forecast_length(self):
        return self.valid_time - self.model_start_time

    @property
    def model_start_time(self):
        return self.forecast_times[self.run_index, 0]

    @property
    def valid_time(self):
        return self.forecast_times[self.run_index, self.forecast_index]

    def next_forecast(self):
        self.forecast_index += 1

    def previous_forecast(self):
        self.forecast_index -= 1

    def next_run(self):
        if self.run_index == len(self.forecast_times) - 1:
            raise IndexError("already at final run")
        self.run_index += 1

    def next_lead_time(self):
        if self.run_index == 0:
            raise IndexError("already at earliest run")
        valid_time = self.valid_time
        forecasts = self.forecast_times[self.run_index - 1].tolist()
        self.forecast_index = forecasts.index(valid_time)
        self.run_index -= 1

    def most_recent_run(self):
        self.run_index = len(self.forecast_times) - 1


def date_range(start, periods, interval):
    return [start + i * interval for i in range(periods)]


class TestTimeControl(unittest.TestCase):
    """Forecast/real time navigation tools"""
    def setUp(self):
        periods = 12
        interval = dt.timedelta(hours=3)
        first_start = dt.datetime(2018, 1, 1, 12, 0, 0)
        first_run = date_range(first_start, periods, interval)
        self.second_start = dt.datetime(2018, 1, 2, 0, 0, 0)
        second_run = date_range(self.second_start, periods, interval)
        forecast_times = [first_run, second_run]
        self.control = TimeControl(forecast_times)

    def test_model_start_time(self):
        result = self.control.model_start_time
        expect = dt.datetime(2018, 1, 1, 12, 0, 0)
        self.assertEqual(expect, result)

    def test_next_forecast_increases_forecast_length(self):
        self.control.next_forecast()
        result = self.control.forecast_length
        expect = dt.timedelta(hours=3)
        self.assertEqual(expect, result)

    def test_previous_forecast_decreases_forecast_length(self):
        self.control.next_forecast()
        self.control.previous_forecast()
        result = self.control.forecast_length
        expect = dt.timedelta(hours=0)
        self.assertEqual(expect, result)

    def test_next_run(self):
        self.control.next_run()
        result = self.control.valid_time
        expect = self.second_start
        self.assertEqual(expect, result)

    def test_next_lead_time_keeps_valid_time_fixed(self):
        self.control.next_run()
        self.control.next_lead_time()
        result = self.control.valid_time
        expect = self.second_start
        self.assertEqual(expect, result)

    def test_next_lead_time_keeps_decrements_model_start_time(self):
        self.control.next_run()
        self.control.next_lead_time()
        result = self.control.model_start_time
        expect = dt.datetime(2018, 1, 1, 12, 0, 0)
        self.assertEqual(expect, result)

    def test_next_lead_time_raises_indexerror_if_on_first_run(self):
        with self.assertRaises(IndexError):
            self.control.next_lead_time()

    def test_next_run_raises_indexerror_if_on_last_run(self):
        self.control.most_recent_run()
        with self.assertRaises(IndexError):
            self.control.next_run()
