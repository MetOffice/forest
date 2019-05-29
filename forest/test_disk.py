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


class TestSearch(unittest.TestCase):
    def test_search(self):
        self.assertTrue(False)


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


class TestGlobalUM(unittest.TestCase):
    def setUp(self):
        self.path = full_path("data/test_global_africa_20190513T0000Z_039.nc")
        self.locator = disk.GlobalUM([self.path])
        self.variables = [
            'air_temperature',
            'dew_point_temperature',
            'geopotential_height',
            'moisture_content_of_soil_layer',
            'precipitation_flux',
            'relative_humidity',
            'relative_humidity_0',
            'relative_humidity_1',
            'soil_moisture_content',
            'surface_temperature',
            'toa_outgoing_longwave_flux',
            'wet_bulb_potential_temperature',
            'x_wind',
            'x_wind_0',
            'y_wind',
            'y_wind_0'
        ]

    def test_stash_variables(self):
        with netCDF4.Dataset(self.path) as dataset:
            result = stash_variables(dataset)
        expect = self.variables
        self.assertEqual(result, expect)

    def test_geopotential_height(self):
        '''
        float geopotential_height(dim0, latitude, longitude) ;
                geopotential_height:standard_name = "geopotential_height" ;
                geopotential_height:units = "m" ;
                geopotential_height:um_stash_source = "m01s16i202" ;
                geopotential_height:grid_mapping = "latitude_longitude" ;
                geopotential_height:coordinates = "forecast_period_1 forecast_reference_time pressure_0 time_1" ;
        '''
        with netCDF4.Dataset(self.path) as dataset:
            var = dataset.variables["time_1"]
            times = netCDF4.num2date(var[:], units=var.units)
            times = np.array(times, dtype='datetime64[s]')
            pressures = dataset.variables['pressure_0'][:]

        variable = "geopotential_height"
        initial = dt.datetime(2019, 5, 13, 0)
        valid = dt.datetime(2019, 5, 14, 15)
        pressure = 1000
        print(np.where(disk.points(times, pressures, valid, pressure)))
        path, pts = self.locator.search(
            variable,
            initial,
            valid,
            pressure)
        self.assertEqual(os.path.basename(path), os.path.basename(self.path))
        np.testing.assert_array_equal(np.where(pts)[0], [48])

    def test_relative_humidity(self):
        variable = "relative_humidity"
        initial = dt.datetime(2019, 5, 13, 0)
        valid = dt.datetime(2019, 5, 14, 15)
        pressure = 1000
        path, pts = self.locator.path_points(variable, initial, valid, pressure)
        self.assertEqual(os.path.basename(path), os.path.basename(self.path))
        self.assertEqual(np.where(pts), ([26],))

    def test_relative_humidity_0(self):
        variable = "relative_humidity_0"
        initial = dt.datetime(2019, 5, 13, 0)
        valid = dt.datetime(2019, 5, 14, 15)
        pressure = 1000
        path, pts = self.locator.path_points(variable, initial, valid, pressure)
        self.assertEqual(os.path.basename(path), os.path.basename(self.path))
        self.assertEqual(np.where(pts), ([23],))

    def test_relative_humidity_1(self):
        variable = "relative_humidity_1"
        initial = dt.datetime(2019, 5, 13, 0)
        valid = dt.datetime(2019, 5, 14, 15)
        pressure = 1000
        path, pts = self.locator.path_points(variable, initial, valid, pressure)
        self.assertEqual(os.path.basename(path), os.path.basename(self.path))
        self.assertEqual(np.where(pts), ([26],))

    def test_soil_moisture_content(self):
        with netCDF4.Dataset(self.path) as dataset:
            result = self.locator._valid_times(dataset, 'soil_moisture_content')
        expect = np.array('2019-05-14 15:00', dtype='datetime64[s]')
        self.assertEqual(result, expect)


@unittest.skip("green light")
class TestIndex(unittest.TestCase):
    def setUp(self):
        self.initial = dt.datetime(2019, 4, 30, 6)
        self.path = full_path("data/test_global_africa_20190513T0000Z_039.nc")
        with netCDF4.Dataset(self.path) as dataset:
            var = dataset.variables["time_1"]
            self.times = netCDF4.num2date(var[:], units=var.units)
            self.pressures = dataset.variables["pressure"][:]

    def test_locate(self):
        time = dt.datetime(2019, 4, 30, 14, 1)
        pressure = 850.
        pts = disk.points(self.times, self.pressures, time, pressure)
        np.testing.assert_array_almost_equal(
                self.pressures[pts], pressure)
        result = self.times[pts][0]
        expect = time.replace(minute=0)
        self.assertEqual(expect, result)

    def test_lengths(self):
        result = disk.lengths(self.times, self.initial)
        self.assertEqual(len(result), 75)
        n = 25
        self.assertEqual(result.tolist(),
                ((n - 1) * [7]) +
                ((n - 1) * [8]) +
                ((n + 2) * [9]))
