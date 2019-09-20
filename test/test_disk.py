import unittest
import datetime as dt
import numpy as np
import netCDF4
import os
import fnmatch
from forest import disk


class TestDateLocator(unittest.TestCase):
    def test_paths_given_date_outside_range_returns_empty_list(self):
        locator = disk.DateLocator([
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
        locator = disk.DateLocator([
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
        locator = disk.DateLocator([
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


class TestNavigator(unittest.TestCase):
    def setUp(self):
        self.path = "test-navigator.nc"

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_given_empty_unified_model_file(self):
        pattern = "*.nc"
        with netCDF4.Dataset(self.path, "w") as dataset:
            pass
        navigator = disk.Navigator([self.path])
        result = navigator.variables(pattern)
        expect = []
        self.assertEqual(expect, result)

    def test_initial_times_given_forecast_reference_time(self):
        pattern = "*.nc"
        with netCDF4.Dataset(self.path, "w") as dataset:
            var = dataset.createVariable("forecast_reference_time", "d", ())
            var.units = "hours since 1970-01-01 00:00:00"
            var[:] = 0
        navigator = disk.Navigator([self.path])
        result = navigator.initial_times(pattern)
        expect = [dt.datetime(1970, 1, 1)]
        self.assertEqual(expect, result)

    def test_valid_times_given_relative_humidity(self):
        pattern = "*.nc"
        variable = "relative_humidity"
        initial_time = dt.datetime(2019, 1, 1)
        valid_times = [
            dt.datetime(2019, 1, 1, 0),
            dt.datetime(2019, 1, 1, 1),
            dt.datetime(2019, 1, 1, 2)
        ]
        pressures = [1000., 1000., 1000.]
        units = "hours since 1970-01-01 00:00:00"
        with netCDF4.Dataset(self.path, "w") as dataset:
            # Dimensions
            dataset.createDimension("dim0", 3)
            dataset.createDimension("longitude", 1)
            dataset.createDimension("latitude", 1)

            # Forecast reference time
            var = dataset.createVariable("forecast_reference_time", "d", ())
            var.units = units
            var[:] = netCDF4.date2num(initial_time, units=units)

            # Time
            var = dataset.createVariable("time", "d", ("dim0",))
            var.units = units
            var[:] = netCDF4.date2num(valid_times, units=units)

            # Pressure
            var = dataset.createVariable("pressure", "d", ("dim0",))
            var[:] = pressures

            # Relative humidity
            var = dataset.createVariable(
                variable, "f", ("dim0", "longitude", "latitude"))
            var[:] = 100.
            var.standard_name = "relative_humidity"
            var.units = "%"
            var.um_stash_source = "m01s16i256"
            var.grid_mapping = "longitude_latitude"
            var.coordinates = "forecast_period forecast_reference_time time"

        navigator = disk.Navigator([self.path])
        result = navigator.valid_times(pattern, variable, initial_time)
        expect = valid_times
        np.testing.assert_array_equal(expect, result)

    def test_pressures_given_relative_humidity(self):
        pattern = "*.nc"
        variable = "relative_humidity"
        initial_time = dt.datetime(2019, 1, 1)
        valid_times = [
            dt.datetime(2019, 1, 1, 0),
            dt.datetime(2019, 1, 1, 1),
            dt.datetime(2019, 1, 1, 2)
        ]
        pressures = [1000., 1000., 1000.]
        units = "hours since 1970-01-01 00:00:00"
        with netCDF4.Dataset(self.path, "w") as dataset:
            # Dimensions
            dataset.createDimension("pressure", 3)
            dataset.createDimension("longitude", 1)
            dataset.createDimension("latitude", 1)

            # Forecast reference time
            var = dataset.createVariable("forecast_reference_time", "d", ())
            var.units = units
            var[:] = netCDF4.date2num(initial_time, units=units)

            # Time
            var = dataset.createVariable("time", "d", ("pressure",))
            var.units = units
            var[:] = netCDF4.date2num(valid_times, units=units)

            # Pressure
            var = dataset.createVariable("pressure", "d", ("pressure",))
            var[:] = pressures

            # Relative humidity
            var = dataset.createVariable(
                variable, "f", ("pressure", "longitude", "latitude"))
            var[:] = 100.
            var.standard_name = "relative_humidity"
            var.units = "%"
            var.um_stash_source = "m01s16i256"
            var.grid_mapping = "longitude_latitude"
            var.coordinates = "forecast_period forecast_reference_time time"

        navigator = disk.Navigator([self.path])
        result = navigator.pressures(pattern, variable, initial_time)
        expect = [1000.]
        np.testing.assert_array_equal(expect, result)


class TestEIDA50(unittest.TestCase):
    def setUp(self):
        self.path = "test-navigate-eida50.nc"
        self.navigator = disk.Navigator([self.path])
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
            self.define(dataset, self.times)
        result = self.navigator.initial_times(self.path)
        expect = [self.times[0]]
        self.assertEqual(expect, result)

    def test_valid_times(self):
        with netCDF4.Dataset(self.path, "w") as dataset:
            self.define(dataset, self.times)
        result = self.navigator.valid_times(
                self.path,
                "toa_brightness_temperature",
                self.times[0])
        expect = self.times
        np.testing.assert_array_equal(expect, result)

    def test_pressures(self):
        with netCDF4.Dataset(self.path, "w") as dataset:
            self.define(dataset, self.times)
        result = self.navigator.pressures(
                self.path,
                "toa_brightness_temperature",
                self.times[0])
        expect = []
        np.testing.assert_array_equal(expect, result)

    def define(self, dataset, times):
        dataset.createDimension("time", len(times))
        dataset.createDimension("longitude", 1)
        dataset.createDimension("latitude", 1)
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
        var[:] = 0
        var = dataset.createVariable(
                "latitude", "f", ("latitude",))
        var.axis = "Y"
        var.units = "degrees_north"
        var.standard_name = "latitude"
        var[:] = 0
        var = dataset.createVariable(
                "data", "f", ("time", "latitude", "longitude"))
        var.standard_name = "toa_brightness_temperature"
        var.long_name = "toa_brightness_temperature"
        var.units = "K"
        var[:] = 0


class TestFNMatch(unittest.TestCase):
    def test_filter(self):
        names = ["/some/file.json", "/other/file.nc"]
        result = fnmatch.filter(names, "*.nc")
        expect = ["/other/file.nc"]
        self.assertEqual(expect, result)


class TestNumpy(unittest.TestCase):
    def test_concatenate(self):
        arrays = [
            np.array([1, 2, 3]),
            np.array([4, 5])
        ]
        result = np.concatenate(arrays)
        expect = np.array([1, 2, 3, 4, 5])
        np.testing.assert_array_equal(expect, result)
