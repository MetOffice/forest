import unittest
from collections import namedtuple
import datetime as dt
import numpy as np
import data




def next_model_run(initial_dates, lengths, current):
    initial_date, length = current
    irun = np.where(np.abs(initial_dates - initial_date) < dt.timedelta(hours=1))[0][0]
    if irun > 0:
        run_gap = hours(initial_dates[irun] - initial_dates[irun - 1])
        return (initial_dates[irun - 1], length + int(run_gap))


def hours(delta):
    return delta.total_seconds() / (60 * 60)


def valid_date(initial, length):
    if not isinstance(length, dt.timedelta):
        length = dt.timedelta(hours=length)
    return initial + length


class TestInitialTimes(unittest.TestCase):
    def test_initial_time(self):
        path = "/some/highway_eakm4p4_20190101T1200Z.nc"
        result = data.initial_time(path)
        expect = dt.datetime(2019, 1, 1, 12)
        self.assertEqual(expect, result)


class TestForecastNavigation(unittest.TestCase):
    def test_next_model_run(self):
        lengths = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24])
        initial_dates = np.array([
            dt.datetime(2019, 1, 1),
            dt.datetime(2019, 1, 1, 12)], dtype=object)
        rd, rl = next_model_run(initial_dates, lengths, (dt.datetime(2019, 1, 1, 12), 0))
        ed, el = (dt.datetime(2019, 1, 1), 12)
        self.assertEqual(ed, rd)
        self.assertEqual(el, rl)

    @unittest.skip("refactoring")
    def test_next_model_run_given_different_times(self):
        lengths = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24])
        initial_dates = np.array([
            dt.datetime(2019, 1, 1, 12),
            dt.datetime(2019, 1, 2),
            dt.datetime(2019, 1, 2, 12)], dtype=object)
        result = next_model_run(initial_dates, lengths, (dt.datetime(2019, 1, 2, 12), 0))
        expect = (dt.datetime(2019, 1, 2), 12)
        self.assertEqual(expect, result)

    @unittest.skip("refactoring")
    def test_next_model_run_given_different_hours(self):
        lengths = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24])
        initial_dates = np.array([
            dt.datetime(2019, 1, 2),
            dt.datetime(2019, 1, 2, 12)], dtype=object)
        result = next_model_run(initial_dates, lengths, (dt.datetime(2019, 1, 1, 12), 0))
        expect = (dt.datetime(2019, 1, 2), 12)
        self.assertEqual(expect, result)
