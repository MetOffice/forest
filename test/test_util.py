import pytest
import cftime
from datetime import datetime
import unittest

import iris
import numpy as np

from forest import util

@pytest.mark.parametrize("given,expect", [
    pytest.param('2019-10-10 01:02:34',
                 datetime(2019, 10, 10, 1, 2, 34),
                 id="str with space"),
    pytest.param('2019-10-10T01:02:34',
                 datetime(2019, 10, 10, 1, 2, 34),
                 id="iso8601"),
    pytest.param(np.datetime64('2019-10-10T11:22:33'),
                 datetime(2019, 10, 10, 11, 22, 33),
                 id="datetime64"),
    pytest.param(cftime.DatetimeGregorian(2019, 10, 10, 11, 22, 33),
                 datetime(2019, 10, 10, 11, 22, 33),
                 id="cftime.DatetimeGregorian"),
])
def test__to_datetime(given, expect):
    assert util.to_datetime(given) == expect


class Test_to_datetime(unittest.TestCase):
    def test_datetime(self):
        dt = datetime.now()
        result = util.to_datetime(dt)
        self.assertEqual(result, dt)

    def test_unsupported(self):
        with self.assertRaisesRegex(Exception, 'Unknown value'):
            util.to_datetime(12)


