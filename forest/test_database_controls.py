import unittest
import unittest.mock
import datetime as dt
import db


class TestControls(unittest.TestCase):
    def setUp(self):
        self.database = db.Database.connect(":memory:")
        self.controls = db.Controls(self.database)

    def tearDown(self):
        self.database.close()

    def test_on_click_emits_state(self):
        key = "k"
        value = "*.nc"
        controls = db.Controls(self.database, patterns=[(key, value)])
        callback = unittest.mock.Mock()
        controls.subscribe(callback)
        controls.on_click('pattern')(value)
        callback.assert_called_once_with(db.State(pattern=value))

    def test_on_variable_emits_state(self):
        value = "token"
        callback = unittest.mock.Mock()
        self.controls.subscribe(callback)
        self.controls.on_click("variable")(value)
        callback.assert_called_once_with(db.State(variable=value))

    @unittest.skip("refactoring test suite")
    def test_on_variable_sets_initial_times_drop_down(self):
        self.controls.on_variable("air_temperature")
        result = self.controls.dropdowns["initial_time"].menu
        expect = [
            ("2019-01-01 00:00:00", "2019-01-01 00:00:00"),
            ("2019-01-01 12:00:00", "2019-01-01 12:00:00"),
        ]
        self.assertEqual(expect, result)

    def test_next_state_given_kwargs(self):
        current = db.State(pattern="p")
        result = db.next_state(current, variable="v")
        expect = db.State(pattern="p", variable="v")
        self.assertEqual(expect, result)

    def test_observable(self):
        state = db.State()
        callback = unittest.mock.Mock()
        self.controls.subscribe(callback)
        self.controls.notify(state)
        callback.assert_called_once_with(state)

    def test_render_state_configures_variable_menu(self):
        self.database.insert_variable("a.nc", "air_temperature")
        self.database.insert_variable("b.nc", "mslp")
        controls = db.Controls(self.database)
        state = db.State(pattern="b.nc")
        controls.render(state)
        result = controls.dropdowns["variable"].menu
        expect = ["mslp"]
        self.assert_label_equal(expect, result)
        self.assert_value_equal(expect, result)

    def test_render_state_configures_initial_time_menu(self):
        for path, time in [
                ("a_0.nc", dt.datetime(2019, 1, 1)),
                ("a_3.nc", dt.datetime(2019, 1, 1, 12))]:
            self.database.insert_file_name(path, time)
        state = db.State(pattern="a_?.nc")
        self.controls.render(state)
        result = self.controls.dropdowns["initial_time"].menu
        expect = ["2019-01-01 00:00:00", "2019-01-01 12:00:00"]
        self.assert_label_equal(expect, result)

    def test_render_given_initial_time_populates_valid_time_menu(self):
        initial = dt.datetime(2019, 1, 1)
        valid = dt.datetime(2019, 1, 1, 3)
        self.database.insert_file_name("file.nc", initial)
        self.database.insert_time("file.nc", "variable", valid, 0)
        state = db.State(initial_time="2019-01-01 00:00:00")
        self.controls.render(state)
        result = self.controls.dropdowns["valid_time"].menu
        expect = ["2019-01-01 03:00:00"]
        self.assert_label_equal(expect, result)

    def test_render_sets_pressure_levels(self):
        initial_time = "2019-01-01 00:00:00"
        pressures = [1000., 950., 850.]
        self.database.insert_file_name("file.nc", initial_time)
        for i, value in enumerate(pressures):
            self.database.insert_pressure("file.nc", "variable", value, i)
        self.controls.render(db.State(initial_time=initial_time))
        result = self.controls.dropdowns["pressure"].menu
        expect = ["1000hPa", "950hPa", "850hPa"]
        self.assert_label_equal(expect, result)

    def test_hpa_given_small_pressures(self):
        result = db.Controls.hpa(0.001)
        expect = "0.001hPa"
        self.assertEqual(expect, result)

    def assert_label_equal(self, expect, result):
        result = [l for l, _ in result]
        self.assertEqual(expect, result)

    def assert_value_equal(self, expect, result):
        result = [v for _, v in result]
        self.assertEqual(expect, result)
