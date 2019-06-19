import unittest
import unittest.mock
import datetime as dt
import db


class TestControls(unittest.TestCase):
    def setUp(self):
        self.database = db.Database.connect(":memory:")
        self.controls = db.Controls(self.database)
        self.blank_message = db.Message('blank', None)

    def tearDown(self):
        self.database.close()

    @unittest.skip("waiting on green light")
    def test_on_change_emits_state(self):
        key = "k"
        value = "*.nc"
        controls = db.Controls(self.database, patterns=[(key, value)])
        callback = unittest.mock.Mock()
        controls.subscribe(callback)
        controls.on_change('pattern')(None, None, value)
        callback.assert_called_once_with(db.State(pattern=value))

    def test_on_variable_emits_state(self):
        value = "token"
        callback = unittest.mock.Mock()
        self.controls.subscribe(callback)
        self.controls.on_change("variable")(None, None, value)
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
        controls = db.Controls(self.database)
        state = db.State(variables=["mslp"])
        controls.render(state)
        result = controls.dropdowns["variable"].menu
        expect = ["mslp"]
        self.assert_label_equal(expect, result)
        self.assert_value_equal(expect, result)

    @unittest.skip("test-driven reducer development")
    def test_send_sets_state_variables(self):
        self.database.insert_variable("a.nc", "air_temperature")
        self.database.insert_variable("b.nc", "mslp")
        controls = db.Controls(self.database)
        controls.state = db.State(pattern="b.nc")
        controls.send(self.blank_message)
        result = controls.state.variables
        expect = ["mslp"]
        self.assertEqual(expect, result)

    def test_render_state_configures_initial_time_menu(self):
        initial_times = ["2019-01-01 12:00:00", "2019-01-01 00:00:00"]
        state = db.State(initial_times=initial_times)
        self.controls.render(state)
        result = self.controls.dropdowns["initial_time"].menu
        expect = initial_times
        self.assert_label_equal(expect, result)

    @unittest.skip("test-driven reducer development")
    def test_send_configures_initial_times(self):
        for path, time in [
                ("a_0.nc", dt.datetime(2019, 1, 1)),
                ("a_3.nc", dt.datetime(2019, 1, 1, 12))]:
            self.database.insert_file_name(path, time)
        self.controls.state = db.State(pattern="a_?.nc")
        self.controls.send(self.blank_message)
        result = self.controls.state.initial_times
        expect = ["2019-01-01 12:00:00", "2019-01-01 00:00:00"]
        self.assertEqual(expect, result)

    def test_render_given_initial_time_populates_valid_time_menu(self):
        initial = dt.datetime(2019, 1, 1)
        valid = dt.datetime(2019, 1, 1, 3)
        self.database.insert_file_name("file.nc", initial)
        self.database.insert_time("file.nc", "variable", valid, 0)
        state = db.State()
        message = db.Message.dropdown("initial_time", "2019-01-01 00:00:00")
        new_state = self.controls.modify(state, message)
        self.controls.render(new_state)
        result = self.controls.dropdowns["valid_time"].menu
        expect = ["2019-01-01 03:00:00"]
        self.assert_label_equal(expect, result)

    def test_render_sets_pressure_levels(self):
        pressures = [1000, 950, 850]
        self.controls.render(db.State(pressures=pressures))
        result = self.controls.dropdowns["pressure"].menu
        expect = ["1000hPa", "950hPa", "850hPa"]
        self.assert_label_equal(expect, result)

    @unittest.skip("test-driven reducer development")
    def test_modify_pressure_levels(self):
        initial_time = "2019-01-01 00:00:00"
        pressures = [1000., 950., 850.]
        self.database.insert_file_name("file.nc", initial_time)
        for i, value in enumerate(pressures):
            self.database.insert_pressure("file.nc", "variable", value, i)
        state = self.controls.modify(
            db.State(initial_time=initial_time),
            self.blank_message)
        result = state.pressures
        expect = pressures
        self.assertEqual(expect, result)

    def test_next_pressure_given_pressures_returns_first_element(self):
        value = 950
        self.controls.state = db.State(pressures=[value])
        self.controls.on_click('pressure', 'next')()
        result = self.controls.state.pressure
        expect = value
        self.assertEqual(expect, result)

    def test_next_pressure_given_pressures_none(self):
        self.controls.state = db.State()
        self.controls.on_click('pressure', 'next')()
        result = self.controls.state.pressure
        expect = None
        self.assertEqual(expect, result)

    def test_next_pressure_given_current_pressure(self):
        pressure = 950
        pressures = [1000, 950, 800]
        self.controls.state = db.State(
            pressures=pressures,
            pressure=pressure)
        self.controls.on_click('pressure', 'next')()
        result = self.controls.state.pressure
        expect = 800
        self.assertEqual(expect, result)

    def test_render_given_pressure(self):
        self.controls.render(db.State(
            pressures=[1000],
            pressure=1000))
        result = self.controls.dropdowns["pressure"].label
        expect = "1000hPa"
        self.assertEqual(expect, result)

    @unittest.skip("waiting on green light")
    def test_database_pressures(self):
        result = self.database.pressures()
        expect = [1000]
        self.assertEqual(expect, result)

    def test_hpa_given_small_pressures(self):
        result = db.Controls.hpa(0.001)
        expect = "0.001hPa"
        self.assertEqual(expect, result)

    def test_render_given_no_variables_disables_dropdown(self):
        self.controls.render(db.State(variables=[]))
        result = self.controls.dropdowns["variable"].disabled
        expect = True
        self.assertEqual(expect, result)

    def test_render_variables_given_none_disables_dropdown(self):
        self.controls.render(db.State(variables=None))
        result = self.controls.dropdowns["variable"].disabled
        expect = True
        self.assertEqual(expect, result)

    def test_render_initial_times_disables_buttons(self):
        key = "initial_time"
        self.controls.render(db.State(initial_times=None))
        self.assertEqual(self.controls.dropdowns[key].disabled, True)
        self.assertEqual(self.controls.buttons[key]["next"].disabled, True)
        self.assertEqual(self.controls.buttons[key]["previous"].disabled, True)

    def test_render_initial_times_enables_buttons(self):
        key = "initial_time"
        self.controls.render(db.State(initial_times=["2019-01-01 00:00:00"]))
        self.assertEqual(self.controls.dropdowns[key].disabled, False)
        self.assertEqual(self.controls.buttons[key]["next"].disabled, False)
        self.assertEqual(self.controls.buttons[key]["previous"].disabled, False)

    def test_render_valid_times_given_none_disables_buttons(self):
        state = db.State(valid_times=None)
        self.check_disabled("valid_time", state, True)

    def test_render_valid_times_given_empty_list_disables_buttons(self):
        state = db.State(valid_times=[])
        self.check_disabled("valid_time", state, True)

    def test_render_valid_times_given_values_enables_buttons(self):
        state = db.State(valid_times=["2019-01-01 00:00:00"])
        self.check_disabled("valid_time", state, False)

    def test_render_pressures_given_none_disables_buttons(self):
        state = db.State(pressures=None)
        self.check_disabled("pressure", state, True)

    def test_render_pressures_given_empty_list_disables_buttons(self):
        state = db.State(pressures=[])
        self.check_disabled("pressure", state, True)

    def test_render_pressures_given_values_enables_buttons(self):
        state = db.State(pressures=[1000.00000001])
        self.check_disabled("pressure", state, False)

    def check_disabled(self, key, state, expect):
        self.controls.render(state)
        self.assertEqual(self.controls.dropdowns[key].disabled, expect)
        self.assertEqual(self.controls.buttons[key]["next"].disabled, expect)
        self.assertEqual(self.controls.buttons[key]["previous"].disabled, expect)

    def assert_label_equal(self, expect, result):
        result = [l for l, _ in result]
        self.assertEqual(expect, result)

    def assert_value_equal(self, expect, result):
        result = [v for _, v in result]
        self.assertEqual(expect, result)


class TestMessage(unittest.TestCase):
    def setUp(self):
        self.database = db.Database.connect(":memory:")

    def tearDown(self):
        self.database.close()

    def test_state_change_given_dropdown_message(self):
        state = db.State()
        message = db.Message.dropdown("pressure", "1000")
        result = db.Controls(self.database).modify(state, message)
        expect = db.State(pressure=1000.)
        self.assertEqual(expect, result)

    def test_state_change_given_previous_initial_time_message(self):
        state = db.State(initial_times=["2019-01-01 00:00:00"])
        message = db.Message.button("initial_time", "previous")
        result = db.Controls(self.database).modify(state, message)
        expect = db.State(
            initial_times=["2019-01-01 00:00:00"],
            initial_time="2019-01-01 00:00:00")
        self.assertEqual(expect.initial_time, result.initial_time)
        self.assertEqual(expect.initial_times, result.initial_times)
        self.assertEqual(expect, result)


class TestNextPrevious(unittest.TestCase):
    def setUp(self):
        self.initial_times = [
            "2019-01-02 00:00:00",
            "2019-01-01 00:00:00",
            "2019-01-04 00:00:00",
            "2019-01-03 00:00:00",
        ]
        self.state = db.State(initial_times=self.initial_times)
        self.store = db.Store(self.state)

    def test_next_given_none_selects_latest_time(self):
        message = db.Message.button("initial_time", "next")
        self.store.dispatch(message)
        result = self.store.state
        expect = db.State(
            initial_time="2019-01-04 00:00:00",
            initial_times=self.initial_times
        )
        self.assert_state_equal(expect, result)

    def test_reducer_next_given_time_moves_forward_in_time(self):
        message = db.Message.button("initial_time", "next")
        state = db.State(
            initial_time="2019-01-01 00:00:00",
            initial_times=[
                "2019-01-01 00:00:00",
                "2019-01-01 02:00:00",
                "2019-01-01 01:00:00",
            ])
        result = db.reducer(state, message)
        expect = db.next_state(state, initial_time="2019-01-01 01:00:00")
        self.assert_state_equal(expect, result)

    def test_previous_given_none_selects_earliest_time(self):
        message = db.Message.button("initial_time", "previous")
        self.store.dispatch(message)
        result = self.store.state
        expect = db.State(
            initial_time="2019-01-01 00:00:00",
            initial_times=self.initial_times
        )
        self.assert_state_equal(expect, result)

    def assert_state_equal(self, expect, result):
        for k, v in expect._asdict().items():
            self.assertEqual(v, getattr(result, k), k)
