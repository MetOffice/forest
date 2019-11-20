import unittest
import os
import datetime as dt
import netCDF4
import numpy as np
from forest import (eida50, satellite, navigate)
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


def _eida50(dataset, times, lons=[0], lats=[0]):
    dataset.createDimension("time", len(times))
    dataset.createDimension("longitude", len(lons))
    dataset.createDimension("latitude", len(lats))
    units = "hours since 1970-01-01 00:00:00"
    var = dataset.createVariable(
            "time", "d", ("time",))
    var.axis = "T"
    var.units = units
    var.standard_name = "time"
    var.calendar = "gregorian"
    var[:] = netCDF4.date2num(times, units=units)
    var = dataset.createVariable(
            "longitude", "f", ("longitude",))
    var.axis = "X"
    var.units = "degrees_east"
    var.standard_name = "longitude"
    var[:] = lons
    var = dataset.createVariable(
            "latitude", "f", ("latitude",))
    var.axis = "Y"
    var.units = "degrees_north"
    var.standard_name = "latitude"
    var[:] = lats
    var = dataset.createVariable(
            "data", "f", ("time", "latitude", "longitude"))
    var.standard_name = "toa_brightness_temperature"
    var.long_name = "toa_brightness_temperature"
    var.units = "K"
    var[:] = 0


class TestCoordinates(unittest.TestCase):
    def setUp(self):
        self.path = "test-navigate-eida50.nc"

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_valid_times_given_eida50_toa_brightness_temperature(self):
        times = [dt.datetime(2019, 1, 1)]
        with netCDF4.Dataset(self.path, "w") as dataset:
            _eida50(dataset, times)

        coord = eida50.Coordinates()
        result = coord.valid_times(self.path, "toa_brightness_temperature")
        expect = times
        self.assertEqual(expect, result)


# Sample data similar to a typical EIDA50 file
TIMES = [dt.datetime(2019, 4, 17) + i * dt.timedelta(minutes=15)
        for i in range(94)]
LONS = np.linspace(-19, 53, 180)  # 10 times fewer for speed
LATS = np.linspace(-13, 23, 90) # 10 times fewer for speed


def test_image(tmpdir):
    path = str(tmpdir / "file_20190417.nc")
    with netCDF4.Dataset(path, "w") as dataset:
        _eida50(dataset, TIMES, LONS, LATS)
    time = dt.datetime(2019, 4, 17, 12)
    loader = satellite.EIDA50(path)
    image = loader.image(time)
    result = set(image.keys())
    expect = set(["x", "y", "dw", "dh", "image"])
    assert expect == result


def test_parse_date():
    path = "/some/EIDA50_takm4p4_20190417.nc"
    result = satellite.Locator.parse_date(path)
    expect = dt.datetime(2019, 4, 17)
    assert expect == result


def test_longitudes(tmpdir):
    path = str(tmpdir / "eida50_20190417.nc")
    with netCDF4.Dataset(path, "w") as dataset:
        _eida50(dataset, TIMES, LONS, LATS)
    loader = satellite.EIDA50(path)
    result = loader.longitudes
    with netCDF4.Dataset(path) as dataset:
        expect = dataset.variables["longitude"][:]
    np.testing.assert_array_almost_equal(expect, result)

def test_latitudes(tmpdir):
    path = str(tmpdir / "eida50_20190417.nc")
    with netCDF4.Dataset(path, "w") as dataset:
        _eida50(dataset, TIMES, LONS, LATS)
    loader = satellite.EIDA50(path)
    result = loader.latitudes
    with netCDF4.Dataset(path) as dataset:
        expect = dataset.variables["latitude"][:]
    np.testing.assert_array_almost_equal(expect, result)

def test_times(tmpdir):
    path = str(tmpdir / "eida50_20190417.nc")
    with netCDF4.Dataset(path, "w") as dataset:
        _eida50(dataset, TIMES, LONS, LATS)
    result = satellite.Locator.load_time_axis(path)
    with netCDF4.Dataset(path) as dataset:
        var = dataset.variables["time"]
        expect = netCDF4.num2date(var[:], units=var.units)
    np.testing.assert_array_equal(expect, result)


class TestEIDA50(unittest.TestCase):
    def setUp(self):
        self.path = "test-navigate-eida50.nc"
        self.navigator = navigate.FileSystemNavigator.from_file_type(
            [self.path], 'eida50')
        self.times = [
            dt.datetime(2019, 1, 1, 0),
            dt.datetime(2019, 1, 1, 0, 15),
            dt.datetime(2019, 1, 1, 0, 30),
            dt.datetime(2019, 1, 1, 0, 45),
        ]

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_initial_times(self):
        with netCDF4.Dataset(self.path, "w") as dataset:
            _eida50(dataset, self.times)
        result = self.navigator.initial_times(self.path)
        expect = [self.times[0]]
        self.assertEqual(expect, result)

    def test_valid_times(self):
        with netCDF4.Dataset(self.path, "w") as dataset:
            _eida50(dataset, self.times)
        result = self.navigator.valid_times(
                self.path,
                "toa_brightness_temperature",
                self.times[0])
        expect = self.times
        np.testing.assert_array_equal(expect, result)

    def test_pressures(self):
        with netCDF4.Dataset(self.path, "w") as dataset:
            _eida50(dataset, self.times)
        result = self.navigator.pressures(
                self.path,
                "toa_brightness_temperature",
                self.times[0])
        expect = []
        np.testing.assert_array_equal(expect, result)
