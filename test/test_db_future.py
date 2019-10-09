import unittest
import os
import netCDF4
import datetime as dt
import sqlite3
import forest.db.future as db


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.database = db.Database(self.connection)

    def tearDown(self):
        self.connection.close()

    def test_connect_to_database_multiple_times(self):
        self.database = db.Database(self.connection)
        self.database = db.Database(self.connection)

    def test_insert_file_name(self):
        names = ["file_1.nc", "file_2.nc"]
        for name in names:
            self.database.insert_file_name(name)
        result = self.database.file_names()
        expect = names
        self.assertEqual(expect, result)

    def test_insert_variable(self):
        for path, variables in [
                ("file_1.nc", ["relative_humidity"]),
                ("file_2.nc", ["mslp", "air_temperature"])]:
            for variable in variables:
                self.database.insert_variable(path, variable)
        result = set(self.database.variables())
        expect = set(["relative_humidity", "mslp", "air_temperature"])
        self.assertEqual(expect, result)

    def test_variables_given_file_pattern(self):
        for path, variables in [
                ("file_1.nc", ["relative_humidity"]),
                ("file_2.nc", ["mslp", "air_temperature"])]:
            for variable in variables:
                self.database.insert_variable(path, variable)
        result = set(self.database.variables(pattern="*2.nc"))
        expect = set(["mslp", "air_temperature"])
        self.assertEqual(expect, result)

    def test_file_names_given_initial_time(self):
        for path, time in [
                ("file_1.nc", dt.datetime(2019, 1, 1)),
                ("file_2.nc", dt.datetime(2019, 1, 2)),
                ("file_3.nc", dt.datetime(2019, 1, 2)),
                ("file_4.nc", dt.datetime(2019, 1, 3))]:
            self.database.insert_file_name(path, initial_time=time)
        result = self.database.file_names(initial_time=dt.datetime(2019, 1, 2))
        expect = ["file_2.nc", "file_3.nc"]
        self.assertEqual(expect, result)

    def test_coordinate(self):
        path = "file.nc"
        variable = "air_temperature"
        names = ["time", "pressure", "longitude", "latitude"]
        for name in names:
            self.database.insert_coordinate(path, variable, name)
        result = self.database.coordinates(path, variable)
        expect = names
        self.assertEqual(expect, result)

    def test_coordinate_support_for_axis(self):
        path = "file.nc"
        variable = "air_temperature"
        names = []
        for name, axis in [
                ("time", 0),
                ("pressure", 0),
                ("forecast_reference_period", None),
                ("longitude", 1),
                ("latitude", 1)]:
            self.database.insert_coordinate(path, variable, name, axis=axis)
        result = self.database.axis(path, variable, "pressure")[0]
        expect = 0
        self.assertEqual(expect, result)

    def test_insert_pressure(self):
        path = "file.nc"
        variable = "air_temperature"
        values = [1000., 950., 850.]
        self.database.insert_pressure(
            path,
            variable,
            values)
        result = self.database.pressures(path, variable)
        expect = [1000., 950., 850.]
        self.assertEqual(expect, result)


class TestInsertNetCDF(unittest.TestCase):
    def setUp(self):
        self.path = "test-insert-netcdf.nc"
        self.define(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_insert_netcdf_given_empty_dim_coords(self):
        database = db.Database.connect(":memory:")
        database.insert_netcdf(self.path)
        result = database.axis(self.path, "air_temperature", "time")
        expect = [None]
        self.assertEqual(expect, result)

    def define(self, path):
        units = "hours since 1970-01-01 00:00:00"
        with netCDF4.Dataset(path, "w") as dataset:
            dataset.createDimension("x", 2)
            dataset.createDimension("y", 2)
            obj = dataset.createVariable("time", "d", ())
            obj[:] = netCDF4.date2num(dt.datetime(2019, 1, 1), units=units)
            obj = dataset.createVariable("x", "f", ("x",))
            obj[:] = [0, 10]
            obj = dataset.createVariable("y", "f", ("y",))
            obj[:] = [0, 10]
            obj = dataset.createVariable("air_temperature", "f", ("y", "x"))
            obj.um_stash_source = "m01s16i203"
            obj.coordinates = "time"
