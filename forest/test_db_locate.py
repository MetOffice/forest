import unittest
import datetime as dt
import db


class TestLocate(unittest.TestCase):
    def test_locate_given_dim0_format_variable(self):
        database = db.Database.connect(":memory:")
        time_axis = 0
        pressure_axis = 0
        for file_name, variable, initial_time in [
                ("file.nc", "air_temperature", dt.datetime(2019, 1, 1))]:
            database.insert_file_name(
                file_name,
                initial_time)
            database.insert_variable(
                file_name,
                variable,
                time_axis,
                pressure_axis)
            for i, value in [
                    (42, dt.datetime(2019, 1, 1, 2))]:
                database.insert_time(file_name, variable, value, i)
            for i, value in [
                    (42, 1000.00001)]:
                database.insert_pressure(file_name, variable, value, i)
        locator = db.Locator(database.connection)
        pattern = "*.nc"
        variable = "air_temperature"
        initial_time = dt.datetime(2019, 1, 1)
        valid_time = dt.datetime(2019, 1, 1, 2)
        pressure = 1000.
        result = locator.locate(
            pattern,
            variable,
            initial_time,
            valid_time,
            pressure
        )
        expect = "file.nc", (42,)
        self.assertEqual(expect, result)

    def test_db_given_axes_0_1_with_pressure_none(self):
        pass
