import unittest
from forest import (
        data,
        db,
        selectors)


class TestDBLoader(unittest.TestCase):
    def setUp(self):
        self.empty_image = {
            "x": [],
            "y": [],
            "dw": [],
            "dh": [],
            "image": [],
            "name": [],
            "units": [],
            "valid": [],
            "initial": [],
            "length": [],
            "level": [],
        }

    def test_image_given_empty_state(self):
        name = None
        pattern = None
        locator = None
        state = db.State()
        selector = selectors.Selector(state)
        loader = data.DBLoader(name, pattern, locator)
        result = loader.image(
                selector.variable,
                selector.initial_time,
                selector.valid_time,
                selector.pressure,
                selector.pressures)
        expect = self.empty_image
        self.assert_dict_equal(expect, result)

    def test_image_given_non_existent_entry_in_database(self):
        name = None
        pattern = None
        database = db.Database.connect(":memory:")
        locator = db.Locator(database.connection)
        state = db.State(
            variable="variable",
            initial_time="2019-01-01 00:00:00",
            valid_time="2019-01-01 00:00:00",
            pressure=1000.)
        selector = selectors.Selector(state)
        loader = data.DBLoader(name, pattern, locator)
        result = loader.image(
                selector.variable,
                selector.initial_time,
                selector.valid_time,
                selector.pressure,
                selector.pressures)
        expect = self.empty_image
        self.assert_dict_equal(expect, result)

    def test_image_given_inconsistent_pressures(self):
        path = "file.nc"
        variable = "variable"
        initial_time = "2019-01-01 00:00:00"
        valid_time = "2019-01-01 00:00:00"
        pressure = 1000.
        database = db.Database.connect(":memory:")
        database.insert_file_name(path, initial_time)
        database.insert_pressure(path, variable, pressure, i=0)
        database.insert_time(path, variable, valid_time, i=0)
        locator = db.Locator(database.connection)
        state = db.State(
            variable=variable,
            initial_time=initial_time,
            valid_time=valid_time,
            pressure=pressure,
            pressures=[925.])
        selector = selectors.Selector(state)
        loader = data.DBLoader(None, "*.nc", locator)
        result = loader.image(
                selector.variable,
                selector.initial_time,
                selector.valid_time,
                selector.pressure,
                selector.pressures)
        expect = self.empty_image
        self.assert_dict_equal(expect, result)

    def assert_dict_equal(self, expect, result):
        self.assertEqual(set(expect.keys()), set(result.keys()))
        for key in expect.keys():
            msg = "values not equal for key='{}'".format(key)
            self.assertEqual(expect[key], result[key], msg)
