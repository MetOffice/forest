import unittest
import datetime as dt
import numpy as np
import netCDF4
import os
import fnmatch
import pytest
import forest.drivers
from forest.drivers import unified_model
from forest import (
        disk,
        tutorial)
from forest.exceptions import SearchFail


class TestLocatorScalability(unittest.TestCase):
    def test_locate_given_one_thousand_files(self):
        N = 10**4
        k = np.random.randint(N)
        start = dt.datetime(2019, 1, 10)
        times = [start + dt.timedelta(hours=i) for i in range(N)]
        paths = ["test-locate-{:%Y%m%dT%H%MZ}.nc".format(time) for time in times]
        locator = unified_model.Locator(paths)
        result = locator.find_paths(times[k])
        expect = [paths[k]]
        self.assertEqual(expect, result)


class TestLocator(unittest.TestCase):
    def setUp(self):
        self.path = "test-navigator.nc"

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_locator_given_empty_file_raises_exception(self):
        pattern = self.path
        with netCDF4.Dataset(self.path, "w") as dataset:
            pass
        variable = "relative_humidity"
        initial_time = dt.datetime(2019, 1, 1)
        valid_time = dt.datetime(2019, 1, 1)
        locator = unified_model.Locator([self.path])
        with self.assertRaises(SearchFail):
            locator.locate(
                    pattern,
                    variable,
                    initial_time,
                    valid_time)

    def test_locator_given_dim0_format(self):
        pattern = self.path
        times = [dt.datetime(2019, 1, 1), dt.datetime(2019, 1, 2)]
        with netCDF4.Dataset(self.path, "w") as dataset:
            um = tutorial.UM(dataset)
            dataset.createDimension("longitude", 1)
            dataset.createDimension("latitude", 1)
            var = um.times("time", length=len(times), dim_name="dim0")
            var[:] = netCDF4.date2num(times, units=var.units)
            um.forecast_reference_time(times[0])
            var = um.pressures("pressure", length=len(times), dim_name="dim0")
            var[:] = 1000.
            dims = ("dim0", "longitude", "latitude")
            coordinates = "forecast_period_1 forecast_reference_time pressure time"
            var = um.relative_humidity(dims, coordinates=coordinates)
            var[:] = 100.
        variable = "relative_humidity"
        initial_time = dt.datetime(2019, 1, 1)
        valid_time = dt.datetime(2019, 1, 2)
        locator = unified_model.Locator([self.path])
        _, result = locator.locate(
                    pattern,
                    variable,
                    initial_time,
                    valid_time,
                    pressure=1000.0001)
        expect = (1,)
        self.assertEqual(expect, result)

    def test_locator_given_time_pressure_format(self):
        pattern = self.path
        reference_time = dt.datetime(2019, 1, 1)
        times = [dt.datetime(2019, 1, 2), dt.datetime(2019, 1, 2, 3)]
        pressures = [1000, 950, 850]
        with netCDF4.Dataset(self.path, "w") as dataset:
            um = tutorial.UM(dataset)
            dataset.createDimension("longitude", 1)
            dataset.createDimension("latitude", 1)
            var = um.times("time", length=len(times))
            var[:] = netCDF4.date2num(times, units=var.units)
            um.forecast_reference_time(reference_time)
            var = um.pressures("pressure", length=len(pressures))
            var[:] = pressures
            dims = ("time", "pressure", "longitude", "latitude")
            coordinates = "forecast_period_1 forecast_reference_time"
            var = um.relative_humidity(dims, coordinates=coordinates)
            var[:] = 100.
        variable = "relative_humidity"
        initial_time = reference_time
        valid_time = times[1]
        pressure = pressures[2]
        locator = unified_model.Locator([self.path])
        _, result = locator.locate(
                    pattern,
                    variable,
                    initial_time,
                    valid_time,
                    pressure)
        expect = (1, 2)
        self.assertEqual(expect, result)

    def test_locator_given_time_outside_time_axis(self):
        pattern = self.path
        reference_time = dt.datetime(2019, 1, 1)
        times = [dt.datetime(2019, 1, 2), dt.datetime(2019, 1, 2, 3)]
        future = dt.datetime(2019, 1, 4)
        pressures = [1000, 950, 850]
        with netCDF4.Dataset(self.path, "w") as dataset:
            um = tutorial.UM(dataset)
            dataset.createDimension("longitude", 1)
            dataset.createDimension("latitude", 1)
            var = um.times("time", length=len(times))
            var[:] = netCDF4.date2num(times, units=var.units)
            um.forecast_reference_time(reference_time)
            var = um.pressures("pressure", length=len(pressures))
            var[:] = pressures
            dims = ("time", "pressure", "longitude", "latitude")
            coordinates = "forecast_period_1 forecast_reference_time"
            var = um.relative_humidity(dims, coordinates=coordinates)
            var[:] = 100.
        variable = "relative_humidity"
        initial_time = reference_time
        valid_time = future
        pressure = pressures[2]
        locator = unified_model.Locator([self.path])
        with self.assertRaises(SearchFail):
            locator.locate(
                    pattern,
                    variable,
                    initial_time,
                    valid_time,
                    pressure)

    def test_initial_time_given_forecast_reference_time(self):
        time = dt.datetime(2019, 1, 1, 12)
        with netCDF4.Dataset(self.path, "w") as dataset:
            um = tutorial.UM(dataset)
            um.forecast_reference_time(time)
        result = unified_model.read_initial_time(self.path)
        expect = time
        np.testing.assert_array_equal(expect, result)

    def test_valid_times(self):
        units = "hours since 1970-01-01 00:00:00"
        times = {
                "time_0": [dt.datetime(2019, 1, 1)],
                "time_1": [dt.datetime(2019, 1, 1, 3)]}
        with netCDF4.Dataset(self.path, "w") as dataset:
            um = tutorial.UM(dataset)
            for name, values in times.items():
                var = um.times(name, length=len(values))
                var[:] = netCDF4.date2num(values, units=var.units)
            var = um.pressures("pressure", length=1)
            var[:] = 1000.
            var = um.longitudes(length=1)
            var[:] = 125.
            var = um.latitudes(length=1)
            var[:] = 45.
            dims = ("time_1", "pressure", "longitude", "latitude")
            var = um.relative_humidity(dims)
            var[:] = 100.
        variable = "relative_humidity"
        result = unified_model.read_valid_times(self.path, variable)
        expect = times["time_1"]
        np.testing.assert_array_equal(expect, result)

    def test_pressure_axis_given_time_pressure_lon_lat_dimensions(self):
        with netCDF4.Dataset(self.path, "w") as dataset:
            um = tutorial.UM(dataset)
            dims = ("time_1", "pressure_0", "longitude", "latitude")
            for dim in dims:
                dataset.createDimension(dim, 1)
            var = um.relative_humidity(dims)
        result = disk.pressure_axis(self.path, "relative_humidity")
        expect = 1
        self.assertEqual(expect, result)

    def test_pressure_axis_given_dim0_format(self):
        coordinates = "forecast_period_1 forecast_reference_time pressure time"
        with netCDF4.Dataset(self.path, "w") as dataset:
            um = tutorial.UM(dataset)
            dims = ("dim0", "longitude", "latitude")
            for dim in dims:
                dataset.createDimension(dim, 1)
            var = um.relative_humidity(dims, coordinates=coordinates)
        result = disk.pressure_axis(self.path, "relative_humidity")
        expect = 0
        self.assertEqual(expect, result)

    def test_time_axis_given_time_pressure_lon_lat_dimensions(self):
        with netCDF4.Dataset(self.path, "w") as dataset:
            um = tutorial.UM(dataset)
            dims = ("time_1", "pressure_0", "longitude", "latitude")
            for dim in dims:
                dataset.createDimension(dim, 1)
            var = um.relative_humidity(dims)
        result = disk.time_axis(self.path, "relative_humidity")
        expect = 0
        self.assertEqual(expect, result)

    def test_time_axis_given_dim0_format(self):
        coordinates = "forecast_period_1 forecast_reference_time pressure time"
        with netCDF4.Dataset(self.path, "w") as dataset:
            um = tutorial.UM(dataset)
            dims = ("dim0", "longitude", "latitude")
            for dim in dims:
                dataset.createDimension(dim, 1)
            var = um.relative_humidity(dims, coordinates=coordinates)
        result = disk.time_axis(self.path, "relative_humidity")
        expect = 0
        self.assertEqual(expect, result)


def test_given_empty_unified_model_file(tmpdir):
    path = str(tmpdir / "file.nc")
    pattern = path
    with netCDF4.Dataset(path, "w") as dataset:
        pass
    settings = {"pattern": path}
    dataset = forest.drivers.get_dataset("unified_model", settings)
    navigator = dataset.navigator()
    result = navigator.variables(pattern)
    expect = []
    assert expect == result


def test_initial_times_given_forecast_reference_time(tmpdir):
    path = str(tmpdir / "file.nc")
    pattern = path
    with netCDF4.Dataset(path, "w") as dataset:
        var = dataset.createVariable("forecast_reference_time", "d", ())
        var.units = "hours since 1970-01-01 00:00:00"
        var[:] = 0
    settings = {"pattern": path}
    dataset = forest.drivers.get_dataset("unified_model", settings)
    navigator = dataset.navigator()
    variable = None
    result = navigator.initial_times(pattern, variable)
    expect = [dt.datetime(1970, 1, 1)]
    assert expect == result


class TestNavigator(unittest.TestCase):
    def setUp(self):
        self.path = "test-navigator.nc"

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

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

        settings = {
            "pattern": self.path
        }
        dataset = forest.drivers.get_dataset("unified_model", settings)
        navigator = dataset.navigator()
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

        settings = {
            "pattern": self.path
        }
        dataset = forest.drivers.get_dataset("unified_model", settings)
        navigator = dataset.navigator()
        result = navigator.pressures(pattern, variable, initial_time)
        expect = [1000.]
        np.testing.assert_array_equal(expect, result)


def test_ndindex_given_dim0_format():
    times = [
        dt.datetime(2019, 1, 1, 0),
        dt.datetime(2019, 1, 1, 3),
        dt.datetime(2019, 1, 1, 6),
        dt.datetime(2019, 1, 1, 0),
        dt.datetime(2019, 1, 1, 3),
        dt.datetime(2019, 1, 1, 6),
    ]
    pressures = [
        1000.0001,
        1000.0001,
        1000.0001,
        0.0001,
        0.0001,
        0.0001,
    ]
    time = times[1]
    pressure = pressures[1]
    masks = [
        disk.time_mask(times, time),
        disk.pressure_mask(pressures, pressure)]
    axes = [0, 0]
    result = disk.ndindex(masks, axes)
    expect = (1,)
    np.testing.assert_array_equal(expect, result)


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

    def test_fancy_indexing(self):
        x = np.zeros((3, 5, 2, 2))
        pts = (0, 2)
        result = x[pts].shape
        expect = (2, 2)
        self.assertEqual(expect, result)
