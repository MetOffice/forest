import unittest
import datetime as dt
import numpy as np
import netCDF4
import os
import fnmatch
import pytest
from forest import (
        disk,
        navigate,
        unified_model)




class UM(object):
    """Unified model diagnostics formatter"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.units = "hours since 1970-01-01 00:00:00"

    def times(self, name, times):
        dataset = self.dataset
        if name not in dataset.dimensions:
            dataset.createDimension(name, len(times))
        var = dataset.createVariable(name, "d", (name,))
        var.axis = "T"
        var.units = self.units
        var.standard_name = "time"
        var.calendar = "gregorian"
        var[:] = netCDF4.date2num(times, units=self.units)

    def forecast_reference_time(self, time, name="forecast_reference_time"):
        dataset = self.dataset
        var = dataset.createVariable(name, "d", ())
        var.units = self.units
        var.standard_name = name
        var.calendar = "gregorian"
        var[:] = netCDF4.date2num(time, units=self.units)

    def pressures(self, length=None, name="pressure"):
        dataset = self.dataset
        if name not in dataset.dimensions:
            dataset.createDimension(name, length)
        var = dataset.createVariable(name, "d", (name,))
        var.axis = "Z"
        var.units = "hPa"
        var.long_name = "pressure"
        return var

    def longitudes(self, length=None, name="longitude"):
        dataset = self.dataset
        if name not in dataset.dimensions:
            dataset.createDimension(name, length)
        var = dataset.createVariable(name, "f", (name,))
        var.axis = "X"
        var.units = "degrees_east"
        var.long_name = "longitude"
        return var

    def latitudes(self, length=None, name="latitude"):
        dataset = self.dataset
        if name not in dataset.dimensions:
            dataset.createDimension(name, length)
        var = dataset.createVariable(name, "f", (name,))
        var.axis = "Y"
        var.units = "degrees_north"
        var.long_name = "latitude"
        return var

    def relative_humidity(self, dims, name="relative_humidity",
            coordinates="forecast_period_1 forecast_reference_time"):
        dataset = self.dataset
        var = dataset.createVariable(name, "f", dims)
        var.standard_name = "relative_humidity"
        var.units = "%"
        var.um_stash_source = "m01s16i204"
        var.grid_mapping = "latitude_longitude"
        var.coordinates = coordinates
        return var


class TestLocator(unittest.TestCase):
    def setUp(self):
        self.path = "test-navigator.nc"

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    @unittest.skip("waiting")
    def test_locator(self):
        pattern = self.path
        with netCDF4.Dataset(self.path, "w") as dataset:
            pass
        variable = "relative_humidity"
        initial_time = dt.datetime(2019, 1, 1)
        valid_time = dt.datetime(2019, 1, 1)
        locator = disk.Locator([self.path])
        result = locator.locate(
                pattern,
                variable,
                initial_time,
                valid_time)
        expect = (self.path, 0)
        self.assertEqual(expect, result)

    def test_initial_time_given_forecast_reference_time(self):
        time = dt.datetime(2019, 1, 1, 12)
        with netCDF4.Dataset(self.path, "w") as dataset:
            um = UM(dataset)
            um.forecast_reference_time(time)
        coords = unified_model.Coordinates()
        result = coords.initial_time(self.path)
        expect = time
        np.testing.assert_array_equal(expect, result)

    def test_valid_times(self):
        units = "hours since 1970-01-01 00:00:00"
        times = {
                "time_0": [dt.datetime(2019, 1, 1)],
                "time_1": [dt.datetime(2019, 1, 1, 3)]}
        with netCDF4.Dataset(self.path, "w") as dataset:
            um = UM(dataset)
            for name, values in times.items():
                um.times(name, values)
            var = um.pressures(length=1)
            var[:] = 1000.
            var = um.longitudes(length=1)
            var[:] = 125.
            var = um.latitudes(length=1)
            var[:] = 45.
            dims = ("time_1", "pressure", "longitude", "latitude")
            var = um.relative_humidity(dims)
            var[:] = 100.
        variable = "relative_humidity"
        coord = unified_model.Coordinates()
        result = coord.valid_times(self.path, variable)
        expect = times["time_1"]
        np.testing.assert_array_equal(expect, result)

    def test_pressure_axis_given_time_pressure_lon_lat_dimensions(self):
        with netCDF4.Dataset(self.path, "w") as dataset:
            um = UM(dataset)
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
            um = UM(dataset)
            dims = ("dim0", "longitude", "latitude")
            for dim in dims:
                dataset.createDimension(dim, 1)
            var = um.relative_humidity(dims, coordinates=coordinates)
        result = disk.pressure_axis(self.path, "relative_humidity")
        expect = 0
        self.assertEqual(expect, result)

    def test_time_axis_given_time_pressure_lon_lat_dimensions(self):
        with netCDF4.Dataset(self.path, "w") as dataset:
            um = UM(dataset)
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
            um = UM(dataset)
            dims = ("dim0", "longitude", "latitude")
            for dim in dims:
                dataset.createDimension(dim, 1)
            var = um.relative_humidity(dims, coordinates=coordinates)
        result = disk.time_axis(self.path, "relative_humidity")
        expect = 0
        self.assertEqual(expect, result)


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
        navigator = navigate.FileSystem.file_type([self.path], "unified_model")
        result = navigator.variables(pattern)
        expect = []
        self.assertEqual(expect, result)

    def test_initial_times_given_forecast_reference_time(self):
        pattern = "*.nc"
        with netCDF4.Dataset(self.path, "w") as dataset:
            var = dataset.createVariable("forecast_reference_time", "d", ())
            var.units = "hours since 1970-01-01 00:00:00"
            var[:] = 0
        navigator = navigate.FileSystem.file_type([self.path], "unified_model")
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

        navigator = navigate.FileSystem([self.path])
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

        navigator = navigate.FileSystem([self.path])
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
