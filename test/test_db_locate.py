import unittest
import sqlite3
import datetime as dt
from forest import db
from forest.exceptions import SearchFail


class TestLocate(unittest.TestCase):
    def setUp(self):
        self.database = db.Database.connect(":memory:")
        self.locator = db.Locator(self.database.connection)

    def test_locate_given_dim0_format_variable(self):
        time_axis = 0
        pressure_axis = 0
        for file_name, variable, initial_time in [
                ("file.nc", "air_temperature", dt.datetime(2019, 1, 1))]:
            self.database.insert_file_name(
                file_name,
                initial_time)
            self.database.insert_variable(
                file_name,
                variable,
                time_axis,
                pressure_axis)
            for i, value in [
                    (42, dt.datetime(2019, 1, 1, 2))]:
                self.database.insert_time(file_name, variable, value, i)
            for i, value in [
                    (42, 1000.00001)]:
                self.database.insert_pressure(file_name, variable, value, i)
        pattern = "*.nc"
        variable = "air_temperature"
        initial_time = dt.datetime(2019, 1, 1)
        valid_time = dt.datetime(2019, 1, 1, 2)
        pressure = 1000.
        result = self.locator.locate(
            pattern,
            variable,
            initial_time,
            valid_time,
            pressure
        )
        expect = "file.nc", (42,)
        self.assertEqual(expect, result)

    def test_locate_given_data_not_in_database_raises_exception(self):
        pattern = "*.nc"
        variable = "air_temperature"
        initial_time = dt.datetime(2019, 1, 1, 0)
        valid_time = dt.datetime(2019, 1, 1, 2)

        with self.assertRaises(SearchFail):
            self.locator.locate(pattern, variable, initial_time, valid_time)


class TestLocator(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.database = db.Database(self.connection)
        self.locator = db.Locator(self.connection)

    def tearDown(self):
        self.connection.close()

    def test_locate_given_surface_criteria(self):
        pattern = "*.nc"
        for path, initial_time in [
                ("a.nc", "2019-01-01 00:00:00"),
                ("b.nc", "2019-01-02 00:00:00")]:
            self.database.insert_file_name(path, initial_time)
            for variable, time, i in [
                    ("mslp", "2019-01-02 00:00:00", 0),
                    ("mslp", "2019-01-02 01:00:00", 1)]:
                self.database.insert_variable(path, variable, time_axis=0)
                self.database.insert_time(path, variable, time, i=i)
        result = self.locator.locate(
            pattern,
            "mslp",
            "2019-01-02 00:00:00",
            "2019-01-02 01:00:00")
        expect = "b.nc", (1,)
        self.assertEqual(expect, result)

    def test_locate(self):
        pattern = "file_*.nc"
        path = "file_000.nc"
        variable = "temperature"
        initial_time = dt.datetime(2019, 1, 1)
        valid_time = dt.datetime(2019, 1, 1, 2)
        pressure = 1000.
        self.database.insert_file_name(path, initial_time)
        self.database.insert_variable(
            path, variable, time_axis=0, pressure_axis=1)
        self.database.insert_time(path, variable, valid_time, i=0)
        self.database.insert_pressure(path, variable, pressure, i=0)
        result = self.locator.locate(
            pattern, variable, initial_time, valid_time, pressure)
        expect = ("file_000.nc", (0, 0))
        self.assertEqual(expect, result)
