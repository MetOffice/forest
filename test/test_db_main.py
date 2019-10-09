import unittest
import datetime as dt
import os
import sqlite3
import netCDF4
import forest.db.main as main
import forest


class TestMain(unittest.TestCase):
    def setUp(self):
        self.units = "hours since 1970-01-01 00:00:00"
        self.database_file = "test_main.db"
        self.netcdf_file = "test_file.nc"
        self._paths = [
            self.database_file,
            self.netcdf_file
        ]

    def tearDown(self):
        for path in self._paths:
            if os.path.exists(path):
                os.remove(path)

    def test_main_writes_file_names_to_database(self):
        with netCDF4.Dataset(self.netcdf_file, "w") as dataset:
            pass

        main.main([
            "--database", self.database_file,
            self.netcdf_file
        ])
        connection = sqlite3.connect(self.database_file)
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM file")
        result = cursor.fetchall()
        expect = [(self.netcdf_file,)]
        self.assertEqual(expect, result)

    def test_main_saves_variable_names_in_database(self):
        with netCDF4.Dataset(self.netcdf_file, "w") as dataset:
            dataset.createDimension("x", 1)
            var = dataset.createVariable("x", "f", ("x",))
            var = dataset.createVariable("air_temperature", "f", ("x",))
            var.um_stash_source = "m01s16i203"
            var = dataset.createVariable("relative_humidity", "f", ("x",))
            var.um_stash_source = "m01s16i256"

        main.main([
            "--database", self.database_file,
            self.netcdf_file
        ])
        connection = sqlite3.connect(self.database_file)
        cursor = connection.cursor()
        cursor.execute("""
                SELECT variable.name, variable.file_id FROM variable
                 ORDER BY variable.name
        """)
        result = cursor.fetchall()
        expect = [("air_temperature", 1), ("relative_humidity", 1)]
        self.assertEqual(expect, result)

    def test_main_saves_times_in_database(self):
        times = [
            dt.datetime(2019, 1, 1, 12),
            dt.datetime(2019, 1, 1, 13)]

        with netCDF4.Dataset(self.netcdf_file, "w") as dataset:
            dataset.createDimension("time", len(times))
            obj = dataset.createVariable("time", "d", ("time",))
            obj.units = self.units
            obj[:] = netCDF4.date2num(times, self.units)
            obj = dataset.createVariable("air_temperature", "f", ("time",))
            obj.um_stash_source = "m01s16i203"

        main.main([
            "--database", self.database_file,
            self.netcdf_file
        ])

        connection = sqlite3.connect(self.database_file)
        cursor = connection.cursor()
        cursor.execute("SELECT DISTINCT value FROM time")
        result = cursor.fetchall()
        expect = [("2019-01-01 12:00:00",), ("2019-01-01 13:00:00",)]
        self.assertEqual(expect, result)

    def test_main_saves_pressure_coordinate(self):
        pressures = [1000.001, 950.001, 800.001]

        with netCDF4.Dataset(self.netcdf_file, "w") as dataset:
            dataset.createDimension("dim0", len(pressures))
            obj = dataset.createVariable("pressure", "d", ("dim0",))
            obj[:] = pressures
            obj = dataset.createVariable("air_temperature", "f", ("dim0",))
            obj.um_stash_source = "m01s16i203"
            obj.coordinates = "pressure"

        main.main([
            "--database", self.database_file,
            self.netcdf_file
        ])

        connection = sqlite3.connect(self.database_file)
        cursor = connection.cursor()
        cursor.execute("SELECT DISTINCT value FROM pressure")
        result = cursor.fetchall()
        expect = [(p,) for p in pressures]
        self.assertEqual(expect, result)

    def test_main_saves_reference_time(self):
        reference_time = dt.datetime(2019, 1, 1)

        with netCDF4.Dataset(self.netcdf_file, "w") as dataset:
            obj = dataset.createVariable("forecast_reference_time", "d", ())
            obj[:] = netCDF4.date2num(reference_time, self.units)
            obj.units = self.units

        main.main([
            "--database", self.database_file,
            self.netcdf_file
        ])

        connection = sqlite3.connect(self.database_file)
        cursor = connection.cursor()
        cursor.execute("SELECT reference FROM file")
        result = cursor.fetchall()
        expect = [(str(reference_time),)]
        self.assertEqual(expect, result)

    def test_main_saves_axis_information(self):
        times = [dt.datetime(2019, 1, 1), dt.datetime(2019, 1, 1)]
        pressures = [1000, 900]
        with netCDF4.Dataset(self.netcdf_file, "w") as dataset:
            dataset.createDimension("dim0", len(times))
            obj = dataset.createVariable("time", "d", ("dim0",))
            obj.units = self.units
            obj[:] = netCDF4.date2num(times, self.units)
            obj = dataset.createVariable("pressure", "d", ("dim0",))
            obj[:] = pressures
            obj = dataset.createVariable("air_temperature", "f", ("dim0",))
            obj.um_stash_source = "m01s16i203"
            obj.coordinates = "time pressure"

        main.main([
            "--database", self.database_file,
            self.netcdf_file
        ])

        connection = sqlite3.connect(self.database_file)
        cursor = connection.cursor()
        cursor.execute("SELECT v.time_axis, v.pressure_axis FROM variable AS v")
        result = cursor.fetchall()
        expect = [(0, 0)]
        self.assertEqual(expect, result)
