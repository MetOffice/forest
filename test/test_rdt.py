import unittest
import datetime as dt
import os
import glob
import json
import numpy as np
from forest import (
        rdt,
        locate)


class TestLocator(unittest.TestCase):
    def setUp(self):
        pattern = os.path.join(os.path.dirname(__file__),
                "sample/RDT*.json")
        self.locator = rdt.Locator(pattern)

    def test_paths(self):
        result = [os.path.basename(path) for path in self.locator.paths]
        expect = ["RDT_features_eastafrica_201904171245.json"]
        self.assertEqual(expect, result)

    def test_find_file(self):
        date = dt.datetime(2019, 4, 17, 12, 59)
        result = os.path.basename(self.locator.find_file(date))
        expect = "RDT_features_eastafrica_201904171245.json"
        self.assertEqual(expect, result)

    def test_parse_date(self):
        path = "/Users/andrewryan/cache/RDT_features_eastafrica_201903151215.json"
        result = self.locator.parse_date(path)
        expect = dt.datetime(2019, 3, 15, 12, 15)
        self.assertEqual(expect, result)

    def test_time_bounds(self):
        dates = [dt.datetime(2019, 8, 1)]
        length = dt.timedelta(minutes=15)
        result = locate.bounds(dates, length)
        expect = np.array([
            ['2019-08-01 00:00:00', '2019-08-01 00:15:00']],
            dtype='datetime64[s]')
        np.testing.assert_array_equal(expect, result)

    def test_in_bounds(self):
        time = '2019-08-01 00:14:59'
        bounds = locate.bounds(['2019-08-01 00:00:00'], dt.timedelta(minutes=15))
        result = locate.in_bounds(bounds, time)
        expect = [True]
        np.testing.assert_array_equal(expect, result)

    def test_in_bounds_given_point_outside_bounds(self):
        time = '2019-08-01 00:15:00'
        bounds = locate.bounds(['2019-08-01 00:00:00'],
                dt.timedelta(minutes=15))
        result = locate.in_bounds(bounds, time)
        expect = [False]
        np.testing.assert_array_equal(expect, result)
