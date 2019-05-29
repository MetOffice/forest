import unittest
import datetime as dt
import numpy as np
import netCDF4
import os
import disk


def full_path(name):
    return os.path.join(os.path.dirname(__file__), name)


def stash_variables(dataset):
    """Find all variables with Stash codes"""
    return [name for name, obj in dataset.variables.items()
            if hasattr(obj, 'um_stash_source')]


def pressures(dataset, name):
    variable = dataset.variables[name]
    dimensions = variable.dimensions
    return dimensions


class TestPattern(unittest.TestCase):
    def test_pattern_given_initial_time_and_length(self):
        initial = np.datetime64('2019-04-29 18:00', 's')
        length = np.timedelta64(33, 'h')
        pattern = "global_africa_{:%Y%m%dT%H%MZ}_umglaa_pa{:03d}.nc"
        result = disk.file_name(pattern, initial, length)
        expect = "global_africa_20190429T1800Z_umglaa_pa033.nc"
        self.assertEqual(expect, result)


class TestLocator(unittest.TestCase):
    def test_paths_given_date_outside_range_returns_empty_list(self):
        locator = disk.Locator([
            "20190101T0000Z.nc",
            "20190101T0600Z.nc",
            "20190101T1200Z.nc",
            "20190101T1800Z.nc",
            "20190102T0000Z.nc"
        ])
        after = np.datetime64('2019-01-02 06:00', 's')
        result = locator.search(after)
        self.assertEqual(result.tolist(), [])

    def test_paths_given_date_in_range_returns_list(self):
        locator = disk.Locator([
            "a/prefix_20190101T0000Z.nc",
            "b/prefix_20190101T0600Z.nc",
            "c/prefix_20190101T1200Z.nc",
            "d/prefix_20190101T1800Z.nc",
            "e/prefix_20190102T0000Z.nc"
        ])
        time = np.datetime64('2019-01-01 18:00', 's')
        result = locator.search(time)
        expect = ["d/prefix_20190101T1800Z.nc"]
        np.testing.assert_array_equal(expect, result)

    def test_paths_given_date_matching_multiple_files(self):
        locator = disk.Locator([
            "a/prefix_20190101T0000Z.nc",
            "b/prefix_20190101T0600Z.nc",
            "c/prefix_20190101T1200Z_000.nc",
            "c/prefix_20190101T1200Z_003.nc",
            "c/prefix_20190101T1200Z_006.nc",
            "c/prefix_20190101T1200Z_009.nc",
            "d/prefix_20190101T1800Z.nc",
            "e/prefix_20190102T0000Z.nc"
        ])
        result = locator.search('2019-01-01 12:00')
        expect = [
            "c/prefix_20190101T1200Z_000.nc",
            "c/prefix_20190101T1200Z_003.nc",
            "c/prefix_20190101T1200Z_006.nc",
            "c/prefix_20190101T1200Z_009.nc",
        ]
        np.testing.assert_array_equal(expect, result)
