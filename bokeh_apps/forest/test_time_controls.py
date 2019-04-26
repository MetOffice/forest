import unittest
from collections import namedtuple
import datetime as dt
import numpy as np
import data
import main


class TestInitialTimes(unittest.TestCase):
    def test_initial_time(self):
        path = "/some/highway_eakm4p4_20190101T1200Z.nc"
        result = data.initial_time(path)
        expect = dt.datetime(2019, 1, 1, 12)
        self.assertEqual(expect, result)


class TestRunControls(unittest.TestCase):
    def test_on_plus_given_no_times(self):
        controls = main.RunControls([])
        controls.on_plus()

    def test_on_minus_given_no_times(self):
        controls = main.RunControls([])
        controls.on_minus()

    def test_on_plus_given_initial_none(self):
        controls = main.RunControls([
                dt.datetime(2019, 1, 1, 12),
                dt.datetime(2019, 1, 1, 13)
            ])
        controls.on_plus()
        self.assertEqual(controls.date, dt.date(2019, 1, 1))
        self.assertEqual(controls.time, dt.time(12, 0))

    def test_on_minus_given_initial_none(self):
        controls = main.RunControls([
                dt.datetime(2019, 1, 1, 12),
                dt.datetime(2019, 1, 1, 13)
            ])
        controls.on_minus()
        self.assertEqual(controls.date, dt.date(2019, 1, 1))
        self.assertEqual(controls.time, dt.time(13, 0))

    def test_on_latest(self):
        dates = [
            dt.datetime(2019, 1, 1),
            dt.datetime(2019, 1, 3, 12),
            dt.datetime(2019, 1, 4, 6),
            dt.datetime(2019, 1, 4, 18),
        ]
        controls = main.RunControls(dates)
        controls.on_latest()
        result = controls.date
        date = dt.date(2019, 1, 4)
        time = dt.time(18)
        self.assertEqual(controls.date, date)
        self.assertEqual(controls.time, time)
