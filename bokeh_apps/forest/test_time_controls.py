import unittest
from collections import namedtuple
import datetime as dt
import numpy as np
import data


class TestInitialTimes(unittest.TestCase):
    def test_initial_time(self):
        path = "/some/highway_eakm4p4_20190101T1200Z.nc"
        result = data.initial_time(path)
        expect = dt.datetime(2019, 1, 1, 12)
        self.assertEqual(expect, result)
