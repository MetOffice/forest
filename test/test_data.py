import unittest
import os
import datetime as dt
import netCDF4
import numpy as np
from forest import (
        satellite,
        data,
        db)


@unittest.skip("green light")
class TestUMLoader(unittest.TestCase):
    def setUp(self):
        self.paths = [
            "/Users/andrewryan/cache/highway_ga6_20190315T0000Z.nc"]
        self.loader = data.UMLoader(self.paths)

    def test_image(self):
        variable = "air_temperature"
        ipressure = 0
        itime = 0
        result = self.loader.image(
                variable,
                ipressure,
                itime)

    def test_find_path(self):
        self.paths = [
            "highway_ga6_20190315T0000Z.nc",
            "highway_ga6_20190315T1200Z.nc"
        ]
        self.finder = data.Finder(self.paths)
        path = self.finder.find(
                dt.datetime(2019, 3, 15, 12))
        result = os.path.basename(path)
        expect = "highway_ga6_20190315T1200Z.nc"
        self.assertEqual(expect, result)


@unittest.skip("green light")
class TestEIDA50(unittest.TestCase):
    def setUp(self):
        self.path = os.path.expanduser("~/cache/EIDA50_takm4p4_20190417.nc")
        self.fixture = satellite.EIDA50([self.path])

    def test_image(self):
        time = dt.datetime(2019, 4, 17, 12)
        image = self.fixture.image(time)
        result = set(image.keys())
        expect = set(["x", "y", "dw", "dh", "image"])
        self.assertEqual(expect, result)

    def test_parse_date(self):
        path = os.path.expanduser("~/cache/EIDA50_takm4p4_20190417.nc")
        result = satellite.EIDA50.parse_date(path)
        expect = dt.datetime(2019, 4, 17)
        self.assertEqual(expect, result)

    def test_longitudes(self):
        result = self.fixture.longitudes
        with netCDF4.Dataset(self.path) as dataset:
            expect = dataset.variables["longitude"][:]
        np.testing.assert_array_almost_equal(expect, result)

    def test_latitudes(self):
        result = self.fixture.latitudes
        with netCDF4.Dataset(self.path) as dataset:
            expect = dataset.variables["latitude"][:]
        np.testing.assert_array_almost_equal(expect, result)

    def test_times(self):
        result = self.fixture.times(self.path)
        with netCDF4.Dataset(self.path) as dataset:
            var = dataset.variables["time"]
            expect = netCDF4.num2date(var[:], units=var.units)
        np.testing.assert_array_equal(expect, result)

    def test_index(self):
        times = np.array([
            dt.datetime(2019, 4, 17, 0),
            dt.datetime(2019, 4, 17, 0, 15),
            dt.datetime(2019, 4, 17, 0, 30)])
        time = dt.datetime(2019, 4, 17, 0, 15)
        result = self.fixture.nearest_index(times, time)
        expect = 1
        self.assertEqual(expect, result)

    def test_index_given_date_between_dates_returns_lower(self):
        times = np.array([
            dt.datetime(2019, 4, 17, 0),
            dt.datetime(2019, 4, 17, 0, 15),
            dt.datetime(2019, 4, 17, 0, 30),
            dt.datetime(2019, 4, 17, 0, 45)])
        time = dt.datetime(2019, 4, 17, 0, 35)
        result = self.fixture.nearest_index(times, time)
        expect = 2
        self.assertEqual(expect, result)


class TestDBLoader(unittest.TestCase):
    def setUp(self):
        self.empty_image = {
            "x": [],
            "y": [],
            "dw": [],
            "dh": [],
            "image": [],
            "name": [],
            "units": [],
            "valid": [],
            "initial": [],
            "length": [],
            "level": [],
        }

    def test_image_given_empty_state(self):
        name = None
        pattern = None
        locator = None
        state = db.State()
        loader = data.DBLoader(name, pattern, locator)
        result = loader.image(state)
        expect = self.empty_image
        self.assert_dict_equal(expect, result)

    def test_image_given_non_existent_entry_in_database(self):
        name = None
        pattern = None
        database = db.Database.connect(":memory:")
        locator = db.Locator(database.connection)
        state = db.State(
            variable="variable",
            initial_time="2019-01-01 00:00:00",
            valid_time="2019-01-01 00:00:00",
            pressure=1000.)
        loader = data.DBLoader(name, pattern, locator)
        result = loader.image(state)
        expect = self.empty_image
        self.assert_dict_equal(expect, result)

    def test_image_given_inconsistent_pressures(self):
        path = "file.nc"
        variable = "variable"
        initial_time = "2019-01-01 00:00:00"
        valid_time = "2019-01-01 00:00:00"
        pressure = 1000.
        database = db.Database.connect(":memory:")
        database.insert_file_name(path, initial_time)
        database.insert_pressure(path, variable, pressure, i=0)
        database.insert_time(path, variable, valid_time, i=0)
        locator = db.Locator(database.connection)
        state = db.State(
            variable=variable,
            initial_time=initial_time,
            valid_time=valid_time,
            pressure=pressure,
            pressures=[925.])
        loader = data.DBLoader(None, "*.nc", locator)
        result = loader.image(state)
        expect = self.empty_image
        self.assert_dict_equal(expect, result)

    @unittest.skip("waiting on database refactor")
    def test_image_given_surface_field(self):
        path = "file.nc"
        variable = "variable"
        initial_time = "2019-01-01 00:00:00"
        valid_time = "2019-01-01 00:00:00"
        pressure = 1000.
        database = db.Database.connect(":memory:")
        database.insert_file_name(path, initial_time)
        database.insert_time(path, variable, valid_time, i=0)
        locator = db.Locator(database.connection)
        state = db.State(
            variable=variable,
            initial_time=initial_time,
            valid_time=valid_time,
            pressure=pressure,
            pressures=[])
        loader = data.DBLoader(None, "*.nc", locator)
        result = loader.image(state)
        expect = {}
        self.assert_dict_equal(expect, result)

    def assert_dict_equal(self, expect, result):
        self.assertEqual(set(expect.keys()), set(result.keys()))
        for key in expect.keys():
            msg = "values not equal for key='{}'".format(key)
            self.assertEqual(expect[key], result[key], msg)
