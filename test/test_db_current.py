import unittest
import datetime as dt
import sqlite3
from forest import db


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.cursor = self.connection.cursor()
        self.database = db.Database(self.connection)
        self.path = "file.nc"
        self.variable = "temperature"

    def tearDown(self):
        self.connection.commit()
        self.connection.close()

    def test_database_opened_twice_on_same_connection(self):
        """Should create table if not exists"""
        db.Database(self.connection)
        db.Database(self.connection)

    def test_file_names_given_no_files_returns_empty_list(self):
        result = self.database.file_names()
        expect = []
        self.assertEqual(result, expect)

    def test_file_names_given_file(self):
        self.database.insert_file_name("file.nc")
        result = self.database.file_names()
        expect = ["file.nc"]
        self.assertEqual(result, expect)

    def test_insert_file_name_unique_constraint(self):
        self.database.insert_file_name("a.nc")
        self.database.insert_file_name("a.nc")
        result = self.database.file_names()
        expect = ["a.nc"]
        self.assertEqual(expect, result)

    def test_insert_variable(self):
        self.database.insert_variable(self.path, self.variable)
        self.cursor.execute("""
            SELECT file.name, variable.name FROM file
              JOIN variable ON file.id = variable.file_id
        """)
        result = self.cursor.fetchall()
        expect = [("file.nc", "temperature")]
        self.assertEqual(expect, result)

    def test_insert_variable_unique_constraint(self):
        self.database.insert_variable(self.path, self.variable)
        self.database.insert_variable(self.path, self.variable)
        self.cursor.execute("SELECT v.name, v.file_id FROM variable AS v")
        result = self.cursor.fetchall()
        expect = [(self.variable, 1)]
        self.assertEqual(expect, result)

    def test_insert_time_unique_constraint(self):
        time = dt.datetime(2019, 1, 1)
        i = 0
        self.database.insert_time(self.path, self.variable, time, i)
        self.database.insert_time(self.path, self.variable, time, i)
        self.cursor.execute("SELECT id, i, value FROM time")
        result = self.cursor.fetchall()
        expect = [(1, i, str(time))]
        self.assertEqual(expect, result)

    def test_insert_pressure_unique_constraint(self):
        pressure = 1000.001
        i = 0
        self.database.insert_pressure(self.path, self.variable, pressure, i)
        self.database.insert_pressure(self.path, self.variable, pressure, i)
        self.cursor.execute("SELECT id, i, value FROM pressure")
        result = self.cursor.fetchall()
        expect = [(1, i, pressure)]
        self.assertEqual(expect, result)

    def test_insert_pressure(self):
        path = "some.nc"
        variable = "relative_humidity"
        i = 18
        pressure = 1000.00001
        self.database.insert_pressure(path, variable, pressure, i)
        result = self.database.find(variable, pressure)
        expect = [("some.nc", i)]
        self.assertEqual(expect, result)

    def test_insert_time_distinguishes_paths(self):
        self.database.insert_time("a.nc", self.variable, "2018-01-01T00:00:00", 6)
        self.database.insert_time("b.nc", self.variable, "2018-01-01T01:00:00", 7)
        result = self.database.find_time(self.variable, "2018-01-01T01:00:00")
        expect = [("b.nc", 7)]
        self.assertEqual(expect, result)

    def test_insert_pressure_distinguishes_paths(self):
        pressure = [1000.0001, 950.0001]
        index = [5, 14]
        path = ["a.nc", "b.nc"]
        self.database.insert_pressure(path[0], self.variable, pressure[0], index[0])
        self.database.insert_pressure(path[1], self.variable, pressure[1], index[1])
        result = self.database.find_pressure(self.variable, pressure[1])
        expect = [(path[1], index[1])]
        self.assertEqual(expect, result)

    def test_insert_times(self):
        path = "file.nc"
        variable = "mslp"
        times = [dt.datetime(2019, 1, 1, 12), dt.datetime(2019, 1, 1, 13)]
        self.database.insert_times(path, variable, times)
        result = self.database.fetch_times(path, variable)
        expect = ["2019-01-01 12:00:00", "2019-01-01 13:00:00"]
        self.assertEqual(expect, result)

    def test_valid_times_returns_all_valid_times(self):
        for (path, variable, time, i) in [
                ("file_0.nc", "var_a", "2019-01-01 00:00:00", 0),
                ("file_1.nc", "var_a", "2019-01-01 01:00:00", 0),
                ("file_1.nc", "var_b", "2019-01-01 02:00:00", 0),
                ("file_2.nc", "var_b", "2019-01-01 03:00:00", 0)]:
            self.database.insert_time(path, variable, time, i)
        result = self.database.valid_times()
        expect = [
            "2019-01-01 00:00:00",
            "2019-01-01 01:00:00",
            "2019-01-01 02:00:00",
            "2019-01-01 03:00:00"]
        self.assertEqual(expect, result)

    def test_valid_times_supports_variable_filtering(self):
        for (path, variable, time, i) in [
                ("file_0.nc", "var_a", "2019-01-01 00:00:00", 0),
                ("file_1.nc", "var_a", "2019-01-01 01:00:00", 0),
                ("file_1.nc", "var_b", "2019-01-01 02:00:00", 0),
                ("file_2.nc", "var_b", "2019-01-01 03:00:00", 0)]:
            self.database.insert_time(path, variable, time, i)
        result = self.database.valid_times(variable="var_b")
        expect = ["2019-01-01 02:00:00", "2019-01-01 03:00:00"]
        self.assertEqual(expect, result)

    def test_valid_times_supports_glob_pattern(self):
        for (path, time, i) in [
                ("file_0.nc", "2019-01-01 00:00:00", 0),
                ("file_1.nc", "2019-01-01 01:00:00", 0),
                ("file_1.nc", "2019-01-01 02:00:00", 0),
                ("file_2.nc", "2019-01-01 03:00:00", 0)]:
            self.database.insert_time(path, self.variable, time, i)
        result = self.database.valid_times(pattern="*_1.nc")
        expect = ["2019-01-01 01:00:00", "2019-01-01 02:00:00"]
        self.assertEqual(expect, result)

    def test_valid_times_given_variable_and_pattern(self):
        for (path, variable, time, i) in [
                ("file_0.nc", "var_a", "2019-01-01 00:00:00", 0),
                ("file_1.nc", "var_a", "2019-01-01 01:00:00", 0),
                ("file_1.nc", "var_b", "2019-01-01 02:00:00", 0),
                ("file_2.nc", "var_b", "2019-01-01 03:00:00", 0)]:
            self.database.insert_time(path, variable, time, i)
        result = self.database.valid_times(
            pattern="*_1.nc",
            variable="var_b")
        expect = ["2019-01-01 02:00:00"]
        self.assertEqual(expect, result)

    def test_valid_times_given_initial_time(self):
        data = [
            ("a.nc", "2019-01-01 00:00:00", [
                "2019-01-01 03:00:00",
                "2019-01-01 06:00:00"]),
            ("b.nc", "2019-01-01 12:00:00", [
                "2019-01-01 12:00:00",
                "2019-01-01 15:00:00",
                "2019-01-01 18:00:00"])
        ]
        for path, initial, times in data:
            self.database.insert_file_name(path, initial)
            for i, time in enumerate(times):
                self.database.insert_time(path, self.variable, time, i)
        result = self.database.valid_times(initial_time="2019-01-01 00:00:00")
        expect = [
            "2019-01-01 03:00:00",
            "2019-01-01 06:00:00"]
        self.assertEqual(expect, result)

    def test_valid_times_given_initial_time_and_variable(self):
        data = [
            ("a.nc", "2019-01-01 00:00:00", [
                ("x", "2019-01-01 03:00:00"),
                ("y", "2019-01-01 06:00:00")]),
            ("b.nc", "2019-01-01 00:00:00", [
                ("x", "2019-01-01 06:00:00"),
                ("y", "2019-01-01 09:00:00")]),
            ("c.nc", "2019-01-01 12:00:00", [
                ("x", "2019-01-01 12:00:00"),
                ("y", "2019-01-01 15:00:00"),
                ("z", "2019-01-01 18:00:00")])
        ]
        for path, initial, items in data:
            self.database.insert_file_name(path, initial)
            for i, (variable, time) in enumerate(items):
                self.database.insert_time(path, variable, time, i)
        result = self.database.valid_times(
            variable="y",
            initial_time="2019-01-01 00:00:00")
        expect = [
            "2019-01-01 06:00:00",
            "2019-01-01 09:00:00"]
        self.assertEqual(expect, result)

    def test_find_all_available_dates(self):
        self.cursor.executemany("""
            INSERT INTO time (i, value) VALUES(:i, :value)
        """, [
            dict(i=0, value="2018-01-01T00:00:00"),
            dict(i=1, value="2018-01-01T01:00:00"),
        ])
        result = self.database.fetch_dates(pattern="a*.nc")
        expect = ["2018-01-01T00:00:00", "2018-01-01T01:00:00"]
        self.assertEqual(expect, result)

    def test_insert_reference_time(self):
        reference_time = dt.datetime(2019, 1, 1, 12)
        self.database.insert_file_name(self.path, reference_time=reference_time)

        self.cursor.execute("SELECT reference FROM file")
        result = self.cursor.fetchall()
        expect = [(str(reference_time),)]
        self.assertEqual(expect, result)

    def test_variable_to_pressure_junction_table_should_be_unique(self):
        pressure = 1000.001
        i = 5

        self.database.insert_pressure(self.path, self.variable, pressure, i)
        self.database.insert_pressure(self.path, self.variable, pressure, i)

        self.cursor.execute("SELECT variable_id,pressure_id FROM variable_to_pressure")
        result = self.cursor.fetchall()
        expect = [(1, 1)]
        self.assertEqual(expect, result)

    def test_variable_to_time_junction_table_should_be_unique(self):
        time = dt.datetime(2019, 1, 1)
        i = 5

        self.database.insert_time(self.path, self.variable, time, i)
        self.database.insert_time(self.path, self.variable, time, i)

        self.cursor.execute("SELECT variable_id,time_id FROM variable_to_time")
        result = self.cursor.fetchall()
        expect = [(1, 1)]
        self.assertEqual(expect, result)

    def test_insert_time_axis_into_variable(self):
        self.database.insert_variable(self.path, self.variable, time_axis=1)
        self.cursor.execute("SELECT time_axis FROM variable")
        result = self.cursor.fetchall()
        expect = [(1,)]
        self.assertEqual(expect, result)

    def test_insert_pressure_axis_into_variable(self):
        self.database.insert_variable(self.path, self.variable, pressure_axis=0)
        self.cursor.execute("SELECT pressure_axis FROM variable")
        result = self.cursor.fetchall()
        expect = [(0,)]
        self.assertEqual(expect, result)

    def test_initial_times_given_empty_database_returns_empty_list(self):
        result = self.database.initial_times()
        expect = []
        self.assertEqual(expect, result)

    def test_initial_times_removes_null_values(self):
        self.database.insert_file_name("file.nc", None)
        result = self.database.initial_times()
        expect = []
        self.assertEqual(expect, result)

    def test_initial_times_given_duplicates_returns_unique_values(self):
        time = dt.datetime(2019, 1, 1)
        self.database.insert_file_name("a.nc", time)
        self.database.insert_file_name("b.nc", time)
        result = self.database.initial_times()
        expect = ["2019-01-01 00:00:00"]
        self.assertEqual(expect, result)

    def test_initial_times_supports_glob_pattern(self):
        self.database.insert_file_name("file_0.nc", "2019-01-01 00:00:00")
        self.database.insert_file_name("file_1.nc", "2019-01-02 00:00:00")
        result = self.database.initial_times(pattern="*_0.nc")
        expect = ["2019-01-01 00:00:00"]
        self.assertEqual(expect, result)

    def test_files(self):
        self.database.insert_file_name("a.nc")
        self.database.insert_file_name("b.nc")
        self.database.insert_file_name("a.nc")
        result = self.database.files()
        expect = ["a.nc", "b.nc"]
        self.assertEqual(expect, result)

    def test_files_supports_glob_pattern(self):
        self.database.insert_file_name("a.nc")
        self.database.insert_file_name("b.nc")
        self.database.insert_file_name("a.nc")
        result = self.database.files(pattern="a.nc")
        expect = ["a.nc"]
        self.assertEqual(expect, result)

    def test_variables(self):
        self.database.insert_variable("a.nc", "var_0")
        self.database.insert_variable("b.nc", "var_1",)
        self.database.insert_variable("a.nc", "var_2")
        result = self.database.variables()
        expect = ["var_0", "var_1", "var_2"]
        self.assertEqual(expect, result)

    def test_variables_given_pattern(self):
        self.database.insert_variable("a.nc", "var_0")
        self.database.insert_variable("b.nc", "var_1",)
        self.database.insert_variable("a.nc", "var_2")
        result = self.database.variables(pattern="a.nc")
        expect = ["var_0", "var_2"]
        self.assertEqual(expect, result)

    def test_pressures(self):
        for (path, variable, pressure, i) in [
                ("file_0.nc", "var_a", 1000, 0),
                ("file_0.nc", "var_a", 950, 1),
                ("file_1.nc", "var_b", 1000, 0),
                ("file_2.nc", "var_b", 1000, 0)]:
            self.database.insert_pressure(path, variable, pressure, i)
        result = self.database.pressures()
        expect = [950, 1000]
        self.assertEqual(expect, result)

    def test_pressures_related_to_variable(self):
        for (path, variable, pressure, i) in [
                ("file_0.nc", "var_a", 1000, 0),
                ("file_0.nc", "var_a", 950, 1),
                ("file_1.nc", "var_b", 800, 0),
                ("file_2.nc", "var_b", 700, 0)]:
            self.database.insert_pressure(path, variable, pressure, i)
        result = self.database.pressures(variable="var_b")
        expect = [700, 800]
        self.assertEqual(expect, result)

    def test_pressures_related_to_pattern(self):
        for (path, variable, pressure, i) in [
                ("file_0.nc", "var_a", 1000, 0),
                ("file_0.nc", "var_a", 950, 1),
                ("file_1.nc", "var_b", 800, 0),
                ("file_2.nc", "var_b", 700, 0)]:
            self.database.insert_pressure(path, variable, pressure, i)
        result = self.database.pressures(pattern="*_2.nc")
        expect = [700]
        self.assertEqual(expect, result)

    def test_pressures_related_to_pattern_and_variable(self):
        for (path, variable, pressure, i) in [
                ("file_0.nc", "var_a", 1000, 0),
                ("file_0.nc", "var_a", 950, 1),
                ("file_0.nc", "var_b", 750, 0),
                ("file_1.nc", "var_b", 800, 0),
                ("file_2.nc", "var_b", 700, 0)]:
            self.database.insert_pressure(path, variable, pressure, i)
        result = self.database.pressures(variable="var_a", pattern="*_0.nc")
        expect = [950., 1000.]
        self.assertEqual(expect, result)

    def test_pressures_related_to_pattern_and_variable_and_initial_time(self):
        for (path, initial_time) in [
                ("file_0.nc", dt.datetime(2019, 1, 1)),
                ("file_1.nc", dt.datetime(2019, 1, 1, 12)),
                ("file_2.nc", dt.datetime(2019, 1, 1, 12))]:
            self.database.insert_file_name(path, initial_time)
        for (path, variable, pressure, i) in [
                ("file_0.nc", "var_a", 1000, 0),
                ("file_0.nc", "var_a", 950, 1),
                ("file_0.nc", "var_b", 750, 0),
                ("file_1.nc", "var_a", 810, 0),
                ("file_1.nc", "var_b", 800, 0),
                ("file_2.nc", "var_b", 700, 0)]:
            self.database.insert_pressure(path, variable, pressure, i)
        result = self.database.pressures(
            variable="var_a",
            pattern="*_[01].nc",
            initial_time=dt.datetime(2019, 1, 1, 12))
        expect = [810.]
        self.assertEqual(expect, result)


class TestCoordinateDB(unittest.TestCase):
    def setUp(self):
        self.database = db.CoordinateDB.connect(":memory:")

    def tearDown(self):
        self.database.close()

    def test_insert_axis(self):
        path = "file.nc"
        variable = "air_temperature"
        coord = "time"
        axis = 0
        self.database.insert_axis(path, variable, coord, axis)
        result = self.database.axis(path, variable, coord)
        expect = axis
        self.assertEqual(expect, result)

    def test_insert_axis_given_mutiple_coordinates(self):
        path = "file.nc"
        variable = "air_temperature"
        for coord, axis in [
                ("time", 0),
                ("pressure", 1),
                ("latitude", 2),
                ("longitude", 3)
            ]:
            self.database.insert_axis(path, variable, coord, axis)
        result = self.database.axis(path, variable, "pressure")
        expect = 1
        self.assertEqual(expect, result)

    def test_insert_axis_given_mutiple_variables(self):
        path = "file.nc"
        for variable, coords in [
                ("mslp", [
                    ("time", 0),
                    ("pressure", 1)]),
                ("relative_humidity", [
                    ("time", 2),
                    ("pressure", 3),
                    ("forecast_period", 4)])
            ]:
            for coord, axis in coords:
                self.database.insert_axis(path, variable, coord, axis)
        result = self.database.axis(path, "relative_humidity", "pressure")
        expect = 3
        self.assertEqual(expect, result)

    def test_insert_axis_given_mutiple_paths(self):
        for path, variable, coords in [
                ("a.nc", "mslp", [
                    ("time", 0),
                    ("pressure", 1)]),
                ("b.nc", "mslp", [
                    ("time", 2),
                    ("pressure", 3),
                    ("forecast_period", 4)])
            ]:
            for coord, axis in coords:
                self.database.insert_axis(path, variable, coord, axis)
        result = self.database.axis("b.nc", "mslp", "time")
        expect = 2
        self.assertEqual(expect, result)

    def test_coordinates_related_to_variable(self):
        for path, variable, coords in [
                ("a.nc", "mslp", [
                    ("time", 0),
                    ("pressure", 1)]),
                ("b.nc", "mslp", [
                    ("time", 2),
                    ("pressure", 3),
                    ("forecast_period", 4)])
            ]:
            for coord, axis in coords:
                self.database.insert_axis(path, variable, coord, axis)
        result = self.database.coordinates("b.nc", "mslp")
        expect = [
            ("time", 2),
            ("pressure", 3),
            ("forecast_period", 4)]
        self.assertEqual(expect, result)

    def test_insert_time_coord(self):
        pattern = "*.nc"
        path = "file.nc"
        variable = "temperature"
        time = dt.datetime(2019, 1, 1)
        times = 3 * [time]
        self.database.insert_times(path, variable, times)
        result = self.database.time_index(pattern, variable, time)
        expect = [0, 1, 2]
        self.assertEqual(expect, result)

    def test_insert_times_related_to_variable(self):
        pattern = "*.nc"
        path = "file.nc"
        for variable, times in [
                ("mslp", [
                    dt.datetime(2019, 1, 1),
                    dt.datetime(2019, 1, 1, 6),
                    dt.datetime(2019, 1, 1, 12)]),
                ("relative_humidity", [
                    dt.datetime(2019, 1, 1, 6),
                    dt.datetime(2019, 1, 1, 12)])]:
            self.database.insert_times(path, variable, times)
        result = self.database.time_index(
            pattern,
            "relative_humidity",
            dt.datetime(2019, 1, 1, 12))
        expect = [1]
        self.assertEqual(expect, result)

    def test_insert_times_related_to_path(self):
        variable = "mslp"
        for path, variable, times in [
                ("a.nc", variable, [
                    dt.datetime(2019, 1, 1),
                    dt.datetime(2019, 1, 1, 6),
                    dt.datetime(2019, 1, 1, 12)]),
                ("b.nc", variable, [
                    dt.datetime(2019, 1, 1, 6),
                    dt.datetime(2019, 1, 1, 12)])]:
            self.database.insert_times(path, variable, times)
        result = self.database.time_index(
            "a.nc",
            variable,
            dt.datetime(2019, 1, 1, 12))
        expect = [2]
        self.assertEqual(expect, result)

    def test_insert_pressures(self):
        path = "file.nc"
        variable = "relative_humidity"
        values = [1., 100., 750., 1000.]
        self.database.insert_pressures(path, variable, values)
        result = self.database.pressure_index(path, variable, 750.)
        expect = [2]
        self.assertEqual(expect, result)

    def test_insert_pressures_given_different_variable(self):
        path = "file.nc"
        self.database.insert_pressures(path, "mslp", [100., 200., 750.])
        self.database.insert_pressures(path, "temperature", [750., 1000.])
        result = self.database.pressure_index(path, "temperature", 750.)
        expect = [0]
        self.assertEqual(expect, result)
