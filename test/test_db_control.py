import unittest
import unittest.mock
import datetime as dt
import numpy as np
from forest import db, redux


def test_convert_datetime64_array_to_strings():
    times = np.array(
            [dt.datetime(2019, 1, 1), dt.datetime(2019, 1, 2)],
            dtype="datetime64[s]")
    result = db.stamps(times)
    expect = ["2019-01-01 00:00:00", "2019-01-02 00:00:00"]
    assert expect == result


def test_type_system_middleware():
    times = np.array(
            [dt.datetime(2019, 1, 1), dt.datetime(2019, 1, 2)],
            dtype="datetime64[s]")
    converter = db.Converter({"valid_times": db.stamps})
    store = redux.Store(db.reducer, middlewares=[converter])
    store.dispatch(db.set_value("valid_times", times))
    result = store.state
    expect = {
        "valid_times": ["2019-01-01 00:00:00", "2019-01-02 00:00:00"]
    }
    assert expect == result


class TestDatabaseMiddleware(unittest.TestCase):
    def setUp(self):
        self.database = db.Database.connect(":memory:")
        self.controls = db.Controls(self.database)
        self.store = redux.Store(db.reducer, middlewares=[self.controls])

    def tearDown(self):
        self.database.close()

    def test_state_change_given_dropdown_message(self):
        action = db.set_value("pressure", "1000")
        self.store.dispatch(action)
        result = self.store.state
        expect = {"pressure": 1000.}
        self.assertEqual(expect, result)

    def test_set_initial_time(self):
        initial = dt.datetime(2019, 1, 1)
        valid = dt.datetime(2019, 1, 1, 3)
        self.database.insert_file_name("file.nc", initial)
        self.database.insert_time("file.nc", "variable", valid, 0)
        action = db.set_value("initial_time", "2019-01-01 00:00:00")
        initial_state={
            "pattern": "file.nc",
            "variable": "variable"}
        store = redux.Store(
            db.reducer,
            initial_state=initial_state,
            middlewares=[self.controls])
        store.dispatch(action)
        result = store.state
        expect = {
            "pattern": "file.nc",
            "variable": "variable",
            "initial_time": str(initial),
            "valid_times": [str(valid)]
        }
        self.assertEqual(expect, result)

    def test_set_pattern(self):
        initial = dt.datetime(2019, 1, 1)
        valid = dt.datetime(2019, 1, 1, 3)
        self.database.insert_file_name("file.nc", initial)
        self.database.insert_time("file.nc", "variable", valid, 0)
        action = db.set_value("pattern", "file.nc")
        self.store.dispatch(action)
        result = self.store.state
        expect = {
            "pattern": "file.nc",
            "variables": ["variable"],
            "initial_times": [str(initial)]
        }
        self.assertEqual(expect, result)

    @unittest.skip("porting to test directory")
    def test_set_valid_time_sets_pressures(self):
        path = "file.nc"
        variable = "air_temperature"
        initial_time = "2019-01-01 00:00:00"
        valid_time = "2019-01-01 12:00:00"
        pressure = 750.
        index = 0
        self.database.insert_file_name(path, initial_time)
        self.database.insert_time(path, variable, valid_time, index)
        self.database.insert_pressure(path, variable, pressure, index)
        store = redux.Store(
            db.reducer,
            middlewares=[self.controls])
        actions = [
            db.set_value("pattern", path),
            db.set_value("variable", variable),
            db.set_value("initial_time", initial_time),
            db.set_value("valid_time", valid_time)]
        for action in actions:
            store.dispatch(action)
        result = store.state
        expect = {
            "pattern": path,
            "variable": variable,
            "variables": [variable],
            "initial_time": initial_time,
            "initial_times": [initial_time],
            "valid_time": valid_time,
            "valid_times": [valid_time],
            "pressures": [pressure],
        }
        self.assertEqual(expect, result)

    @unittest.skip("porting to test directory")
    def test_set_variable_given_initial_time_changes_times_and_pressure(self):
        path = "some.nc"
        initial_time = "2019-01-01 00:00:00"
        self.database.insert_file_name(path, initial_time)
        index = 0
        for variable, valid_time, pressure in [
                ("air_temperature", "2019-01-01 01:00:00", 1000.),
                ("olr", "2019-01-01 01:03:00", 10.)]:
            self.database.insert_time(path, variable, valid_time, index)
            self.database.insert_pressure(path, variable, pressure, index)

        store = redux.Store(
            db.reducer,
            middlewares=[self.controls])
        actions = [
            db.set_value("pattern", path),
            db.set_value("variable", "air_temperature"),
            db.set_value("initial_time", initial_time),
            db.set_value("variable", "olr")
        ]
        for action in actions:
            store.dispatch(action)

        result = store.state
        expect = {
            "pattern": path,
            "variable": "olr",
            "variables": ["air_temperature", "olr"],
            "initial_time": initial_time,
            "initial_times": [initial_time],
            "valid_times": ["2019-01-01 01:03:00"],
            "pressures": [0.0]
        }
        self.assertEqual(expect, result)

    @unittest.skip("porting to test directory")
    def test_navigator_api(self):
        path = "file.nc"
        variable = "air_temperature"
        initial_time = "2019-01-01 00:00:00"
        valid_time = "2019-01-01 12:00:00"
        pressure = 750.
        navigator = db.Navigator()
        controls = db.Controls(navigator)
        store = redux.Store(
            db.reducer,
            middlewares=[controls])
        actions = [
            db.set_value("pattern", path),
            db.set_value("variable", variable),
            db.set_value("initial_time", initial_time),
            db.set_value("valid_time", valid_time)]
        for action in actions:
            store.dispatch(action)
        result = store.state
        expect = {
            "pattern": path,
            "variable": variable,
            "variables": [variable],
            "initial_time": initial_time,
            "initial_times": [initial_time],
            "valid_time": valid_time,
            "valid_times": [valid_time],
            "pressures": [pressure],
        }
        self.assertEqual(expect, result)


class TestControls(unittest.TestCase):
    def setUp(self):
        self.database = db.Database.connect(":memory:")

    def tearDown(self):
        self.database.close()

    def test_on_variable_emits_state(self):
        value = "token"
        listener = unittest.mock.Mock()
        view = db.ControlView()
        view.subscribe(listener)
        view.on_change("variable")(None, None, value)
        listener.assert_called_once_with(db.set_value("variable", value))

    def test_next_pressure_given_pressures_returns_first_element(self):
        pressure = 950
        store = redux.Store(
            db.reducer,
            initial_state={"pressures": [pressure]},
            middlewares=[
                db.next_previous,
                db.Controls(self.database)])
        view = db.ControlView()
        view.subscribe(store.dispatch)
        view.on_next('pressure', 'pressures')()
        result = store.state
        expect = {
            "pressure": pressure,
            "pressures": [pressure]
        }
        self.assertEqual(expect, result)

    def test_next_pressure_given_pressures_none(self):
        store = redux.Store(
            db.reducer,
            middlewares=[
                db.InverseCoordinate("pressure"),
                db.next_previous,
                db.Controls(self.database)
            ])
        view = db.ControlView()
        view.subscribe(store.dispatch)
        view.on_next('pressure', 'pressures')()
        result = store.state
        expect = {}
        self.assertEqual(expect, result)

    def test_next_pressure_given_current_pressure(self):
        pressure = 950
        pressures = [1000, 950, 800]
        store = redux.Store(
            db.reducer,
            initial_state={
                "pressure": pressure,
                "pressures": pressures
            },
            middlewares=[
                db.InverseCoordinate("pressure"),
                db.next_previous,
                db.Controls(self.database)
            ])
        view = db.ControlView()
        view.subscribe(store.dispatch)
        view.on_next('pressure', 'pressures')()
        result = store.state["pressure"]
        expect = 800
        self.assertEqual(expect, result)


class TestControlView(unittest.TestCase):
    def setUp(self):
        self.view = db.ControlView()

    def test_on_click_emits_action(self):
        listener = unittest.mock.Mock()
        self.view.subscribe(listener)
        self.view.on_next("pressure", "pressures")()
        expect = db.next_value("pressure", "pressures")
        listener.assert_called_once_with(expect)

    def test_render_given_no_variables_disables_dropdown(self):
        self.view.render({"variables": []})
        result = self.view.dropdowns["variable"].disabled
        expect = True
        self.assertEqual(expect, result)

    def test_render_given_pressure(self):
        state = {
            "pressures": [1000],
            "pressure": 1000
        }
        self.view.render(state)
        result = self.view.dropdowns["pressure"].label
        expect = "1000hPa"
        self.assertEqual(expect, result)

    def test_render_sets_pressure_levels(self):
        pressures = [1000, 950, 850]
        self.view.render({"pressures": pressures})
        result = self.view.dropdowns["pressure"].menu
        expect = ["1000hPa", "950hPa", "850hPa"]
        self.assert_label_equal(expect, result)

    def test_render_given_initial_time_populates_valid_time_menu(self):
        state = {"valid_times": [dt.datetime(2019, 1, 1, 3)]}
        self.view.render(state)
        result = self.view.dropdowns["valid_time"].menu
        expect = ["2019-01-01 03:00:00"]
        self.assert_label_equal(expect, result)

    def test_render_state_configures_variable_menu(self):
        self.view.render({"variables": ["mslp"]})
        result = self.view.dropdowns["variable"].menu
        expect = ["mslp"]
        self.assert_label_equal(expect, result)
        self.assert_value_equal(expect, result)

    def test_render_state_configures_initial_time_menu(self):
        initial_times = ["2019-01-01 12:00:00", "2019-01-01 00:00:00"]
        state = {"initial_times": initial_times}
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
        self.view.render({})
        self.assertEqual(self.view.dropdowns[key].disabled, True)
        self.assertEqual(self.view.buttons[key]["next"].disabled, True)
        self.assertEqual(self.view.buttons[key]["previous"].disabled, True)

    def test_hpa_given_small_pressures(self):
        result = db.ControlView.hpa(0.001)
        expect = "0.001hPa"
        self.assertEqual(expect, result)

    def test_render_variables_given_null_state_disables_dropdown(self):
        self.view.render({})
        result = self.view.dropdowns["variable"].disabled
        expect = True
        self.assertEqual(expect, result)

    def test_render_initial_times_enables_buttons(self):
        key = "initial_time"
        self.view.render({"initial_times": ["2019-01-01 00:00:00"]})
        self.assertEqual(self.view.dropdowns[key].disabled, False)
        self.assertEqual(self.view.buttons[key]["next"].disabled, False)
        self.assertEqual(self.view.buttons[key]["previous"].disabled, False)

    def test_render_valid_times_given_null_state_disables_buttons(self):
        self.check_disabled("valid_time", {}, True)

    def test_render_valid_times_given_empty_list_disables_buttons(self):
        self.check_disabled("valid_time", {"valid_times": []}, True)

    def test_render_valid_times_given_values_enables_buttons(self):
        state = {"valid_times": ["2019-01-01 00:00:00"]}
        self.check_disabled("valid_time", state, False)

    def test_render_pressures_given_null_state_disables_buttons(self):
        self.check_disabled("pressure", {}, True)

    def test_render_pressures_given_empty_list_disables_buttons(self):
        self.check_disabled("pressure", {"pressures": []}, True)

    def test_render_pressures_given_values_enables_buttons(self):
        self.check_disabled("pressure", {"pressures": [1000.00000001]}, False)

    def check_disabled(self, key, state, expect):
        self.view.render(state)
        self.assertEqual(self.view.dropdowns[key].disabled, expect)
        self.assertEqual(self.view.buttons[key]["next"].disabled, expect)
        self.assertEqual(self.view.buttons[key]["previous"].disabled, expect)


class TestNextPrevious(unittest.TestCase):
    def setUp(self):
        self.initial_times = [
            "2019-01-02 00:00:00",
            "2019-01-01 00:00:00",
            "2019-01-04 00:00:00",
            "2019-01-03 00:00:00",
        ]

    def test_middleware_converts_next_value_to_set_value(self):
        log = db.Log()
        state = {
            "k": 2,
            "ks": [1, 2, 3]
        }
        store = redux.Store(
            db.reducer,
            initial_state=state,
            middlewares=[
                db.next_previous,
                log])
        store.dispatch(db.next_value("k", "ks"))
        result = store.state
        expect = {
            "k": 3,
            "ks": [1, 2, 3]
        }
        self.assertEqual(expect, result)
        self.assertEqual(log.actions, [db.set_value("k", 3)])

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
        store = redux.Store(
            db.reducer,
            initial_state=state,
            middlewares=[db.next_previous])
        store.dispatch(action)
        result = store.state
        expect = dict(
            initial_time="2019-01-04 00:00:00",
            initial_times=self.initial_times)
        self.assertEqual(expect, result)

    def test_reducer_given_set_value_action_adds_key_value(self):
        action = db.set_value("name", "value")
        state = {"previous": "constant"}
        result = db.reducer(state, action)
        expect = {
            "previous": "constant",
            "name": "value"
        }
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
        store = redux.Store(
            db.reducer,
            initial_state=state,
            middlewares=[db.next_previous])
        store.dispatch(action)
        result = store.state
        expect = {
            "initial_time": "2019-01-01 01:00:00",
            "initial_times": initial_times
        }
        self.assertEqual(expect, result)

    def test_previous_given_none_selects_earliest_time(self):
        action = db.previous_value("initial_time", "initial_times")
        state = {"initial_times": self.initial_times}
        store = redux.Store(
            db.reducer,
            initial_state=state,
            middlewares=[db.next_previous])
        store.dispatch(action)
        result = store.state
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


class TestPressureMiddleware(unittest.TestCase):
    def test_pressure_middleware_reverses_pressure_coordinate(self):
        pressures = [1000, 850, 500]
        state = {
            "pressure": 850,
            "pressures": pressures
        }
        store = redux.Store(
            db.reducer,
            initial_state=state,
            middlewares=[
                db.InverseCoordinate("pressure"),
                db.next_previous])
        action = db.next_value("pressure", "pressures")
        store.dispatch(action)
        result = store.state
        expect = {
            "pressure": 500,
            "pressures": pressures
        }
        self.assertEqual(expect, result)


class TestStateStream(unittest.TestCase):
    def test_support_old_style_state(self):
        """Not all components are ready to accept dict() states"""
        listener = unittest.mock.Mock()
        store = redux.Store(db.reducer)
        old_states = (db.Stream()
                  .listen_to(store)
                  .map(lambda x: db.State(**x)))
        old_states.subscribe(listener)
        store.dispatch(db.set_value("pressure", 1000))
        expect = db.State(pressure=1000)
        listener.assert_called_once_with(expect)
