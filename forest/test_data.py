import unittest
import os
import datetime as dt
import netCDF4
import numpy as np
import satellite
import data


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
