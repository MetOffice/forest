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

    def test_on_variable_emits_state(self):
        value = "token"
        callback = unittest.mock.Mock()
        self.controls.subscribe(callback)
        self.controls.view.on_change("variable")(None, None, value)
        callback.assert_called_once_with(db.State(variable=value))

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

    def test_middleware_database_sets_initial_time(self):
        initial = dt.datetime(2019, 1, 1)
        valid = dt.datetime(2019, 1, 1, 3)
        self.database.insert_file_name("file.nc", initial)
        self.database.insert_time("file.nc", "variable", valid, 0)
        state = db.State()
        message = db.Message.dropdown("initial_time", "2019-01-01 00:00:00")
        state = self.controls.modify(state, message)
        self.assertEqual(state.initial_time, str(initial))
        self.assertEqual(state.valid_times, [str(valid)])

    def test_next_pressure_given_pressures_returns_first_element(self):
        value = 950
        controls = db.Controls(
            self.database,
            state=db.State(pressures=[value]))
        controls.view.on_click('pressure', 'next')()
        result = controls.state.pressure
        expect = value
        self.assertEqual(expect, result)

    def test_next_pressure_given_pressures_none(self):
        controls = db.Controls(
            self.database,
            state=db.State())
        controls.view.on_click('pressure', 'next')()
        result = controls.state.pressure
        expect = None
        self.assertEqual(expect, result)

    def test_next_pressure_given_current_pressure(self):
        pressure = 950
        pressures = [1000, 950, 800]
        controls = db.Controls(
            self.database,
            state=db.State(
                pressures=pressures,
                pressure=pressure)
        )
        controls.view.on_click('pressure', 'next')()
        result = controls.state.pressure
        expect = 800
        self.assertEqual(expect, result)


class TestControlView(unittest.TestCase):
    def setUp(self):
        self.view = db.ControlView()

    def test_render_given_no_variables_disables_dropdown(self):
        self.view.render(db.State(variables=[]))
        result = self.view.dropdowns["variable"].disabled
        expect = True
        self.assertEqual(expect, result)

    def test_render_given_pressure(self):
        self.view.render(db.State(
            pressures=[1000],
            pressure=1000))
        result = self.view.dropdowns["pressure"].label
        expect = "1000hPa"
        self.assertEqual(expect, result)

    def test_render_sets_pressure_levels(self):
        pressures = [1000, 950, 850]
        self.view.render(db.State(pressures=pressures))
        result = self.view.dropdowns["pressure"].menu
        expect = ["1000hPa", "950hPa", "850hPa"]
        self.assert_label_equal(expect, result)

    def test_render_given_initial_time_populates_valid_time_menu(self):
        state = db.State(valid_times=[dt.datetime(2019, 1, 1, 3)])
        self.view.render(state)
        result = self.view.dropdowns["valid_time"].menu
        expect = ["2019-01-01 03:00:00"]
        self.assert_label_equal(expect, result)

    def test_render_state_configures_variable_menu(self):
        state = db.State(variables=["mslp"])
        self.view.render(state)
        result = self.view.dropdowns["variable"].menu
        expect = ["mslp"]
        self.assert_label_equal(expect, result)
        self.assert_value_equal(expect, result)

    def test_render_state_configures_initial_time_menu(self):
        initial_times = ["2019-01-01 12:00:00", "2019-01-01 00:00:00"]
        state = db.State(initial_times=initial_times)
        self.view.render(state)
        result = self.view.dropdowns["initial_time"].menu
        expect = initial_times
        self.assert_label_equal(expect, result)

    def assert_label_equal(self, expect, result):
        result = [l for l, _ in result]
        self.assertEqual(expect, result)

    def assert_value_equal(self, expect, result):
        result = [v for _, v in result]
        self.assertEqual(expect, result)

    def test_render_initial_times_disables_buttons(self):
        key = "initial_time"
        self.view.render(db.State(initial_times=None))
        self.assertEqual(self.view.dropdowns[key].disabled, True)
        self.assertEqual(self.view.buttons[key]["next"].disabled, True)
        self.assertEqual(self.view.buttons[key]["previous"].disabled, True)

    def test_hpa_given_small_pressures(self):
        result = db.ControlView.hpa(0.001)
        expect = "0.001hPa"
        self.assertEqual(expect, result)

    def test_render_variables_given_none_disables_dropdown(self):
        self.view.render(db.State(variables=None))
        result = self.view.dropdowns["variable"].disabled
        expect = True
        self.assertEqual(expect, result)

    def test_render_initial_times_enables_buttons(self):
        key = "initial_time"
        self.view.render(db.State(initial_times=["2019-01-01 00:00:00"]))
        self.assertEqual(self.view.dropdowns[key].disabled, False)
        self.assertEqual(self.view.buttons[key]["next"].disabled, False)
        self.assertEqual(self.view.buttons[key]["previous"].disabled, False)

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
        self.view.render(state)
        self.assertEqual(self.view.dropdowns[key].disabled, expect)
        self.assertEqual(self.view.buttons[key]["next"].disabled, expect)
        self.assertEqual(self.view.buttons[key]["previous"].disabled, expect)


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

    def test_next_value_action_creator(self):
        result = db.next_value("initial_time", "initial_times")
        expect = {
            "kind": "NEXT_VALUE",
            "payload": {
                "item_key": "initial_time",
                "items_key": "initial_times"
            }
        }
        self.assertEqual(expect, result)

    def test_previous_value_action_creator(self):
        result = db.previous_value("initial_time", "initial_times")
        expect = {
            "kind": "PREVIOUS_VALUE",
            "payload": {
                "item_key": "initial_time",
                "items_key": "initial_times"
            }
        }
        self.assertEqual(expect, result)

    def test_set_value_action_creator(self):
        result = db.set_value("K", "V")
        expect = {
            "kind": "SET_VALUE",
            "payload": dict(key="K", value="V")
        }
        self.assertEqual(expect, result)

    def test_next_given_none_selects_latest_time(self):
        action = db.next_value("initial_time", "initial_times")
        state = dict(initial_times=self.initial_times)
        result = db.reducer(state, action)
        expect = dict(
            initial_time="2019-01-04 00:00:00",
            initial_times=self.initial_times)
        self.assertEqual(expect, result)

    def test_reducer_next_given_time_moves_forward_in_time(self):
        initial_times = [
            "2019-01-01 00:00:00",
            "2019-01-01 02:00:00",
            "2019-01-01 01:00:00"
        ]
        action = db.next_value("initial_time", "initial_times")
        state = {
            "initial_time": "2019-01-01 00:00:00",
            "initial_times": initial_times
        }
        result = db.reducer(state, action)
        expect = {
            "initial_time": "2019-01-01 01:00:00",
            "initial_times": initial_times
        }
        self.assertEqual(expect, result)

    def test_previous_given_none_selects_earliest_time(self):
        action = db.previous_value("initial_time", "initial_times")
        state = {"initial_times": self.initial_times}
        result = db.reducer(state, action)
        expect = {
            "initial_time": "2019-01-01 00:00:00",
            "initial_times": self.initial_times
        }
        self.assertEqual(expect, result)

    def test_next_item_given_last_item_returns_first_item(self):
        result = db.control.next_item([0, 1, 2], 2)
        expect = 0
        self.assertEqual(expect, result)

    def test_previous_item_given_first_item_returns_last_item(self):
        result = db.control.previous_item([0, 1, 2], 0)
        expect = 2
        self.assertEqual(expect, result)
