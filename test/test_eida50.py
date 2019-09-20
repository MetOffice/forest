import unittest
import os
import datetime as dt
import netCDF4
from forest import satellite
from forest.exceptions import FileNotFound, IndexNotFound


class TestLocator(unittest.TestCase):
    def setUp(self):
        self.paths = []
        self.pattern = "test-eida50*.nc"
        self.locator = satellite.Locator(self.pattern)

    def tearDown(self):
        for path in self.paths:
            if os.path.exists(path):
                os.remove(path)

    def test_parse_date(self):
        path = "/some/file-20190101.nc"
        result = self.locator.parse_date(path)
        expect = dt.datetime(2019, 1, 1)
        self.assertEqual(expect, result)

    def test_find_given_no_files_raises_notfound(self):
        any_date = dt.datetime.now()
        with self.assertRaises(FileNotFound):
            self.locator.find(any_date)

    def test_find_given_a_single_file(self):
        valid_date = dt.datetime(2019, 1, 1)
        path = "test-eida50-20190101.nc"
        self.paths.append(path)

        times = [valid_date]
        with netCDF4.Dataset(path, "w") as dataset:
            self.set_times(dataset, times)

        found_path, index = self.locator.find(valid_date)
        self.assertEqual(found_path, path)
        self.assertEqual(index, 0)

    def test_find_given_multiple_files(self):
        dates = [
                dt.datetime(2019, 1, 1),
                dt.datetime(2019, 1, 2),
                dt.datetime(2019, 1, 3)]
        for date in dates:
            path = "test-eida50-{:%Y%m%d}.nc".format(date)
            self.paths.append(path)
            with netCDF4.Dataset(path, "w") as dataset:
                self.set_times(dataset, [date])
        valid_date = dt.datetime(2019, 1, 2, 0, 14)
        found_path, index = self.locator.find(valid_date)
        expect_path = "test-eida50-20190102.nc"
        self.assertEqual(found_path, expect_path)
        self.assertEqual(index, 0)

    def test_find_index_given_valid_time(self):
        time = dt.datetime(2019, 1, 1, 3, 31)
        times = [
            dt.datetime(2019, 1, 1, 3, 0),
            dt.datetime(2019, 1, 1, 3, 15),
            dt.datetime(2019, 1, 1, 3, 30),
            dt.datetime(2019, 1, 1, 3, 45),
            dt.datetime(2019, 1, 1, 4, 0),
        ]
        freq = dt.timedelta(minutes=15)
        result = self.locator.find_index(times, time, freq)
        expect = 2
        self.assertEqual(expect, result)

    def test_find_index_outside_range_raises_exception(self):
        time = dt.datetime(2019, 1, 4, 16)
        times = [
            dt.datetime(2019, 1, 1, 3, 0),
            dt.datetime(2019, 1, 1, 3, 15),
            dt.datetime(2019, 1, 1, 3, 30),
            dt.datetime(2019, 1, 1, 3, 45),
            dt.datetime(2019, 1, 1, 4, 0),
        ]
        freq = dt.timedelta(minutes=15)
        with self.assertRaises(IndexNotFound):
            self.locator.find_index(times, time, freq)

    def set_times(self, dataset, times):
        units = "seconds since 1970-01-01 00:00:00"
        dataset.createDimension("time", len(times))
        var = dataset.createVariable("time", "d", ("time",))
        var.units = units
        var[:] = netCDF4.date2num(times, units=units)
