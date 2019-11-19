import unittest
import datetime as dt
import netCDF4
import numpy as np
from forest import (
        data,
        db,
        unified_model,
        selectors)


def _um_file(dataset, times, pressures, lats, lons):
    units = "hours since 1970-01-01 00:00:00"
    dataset.createDimension("longitude", len(lons))
    dataset.createDimension("latitude", len(lats))
    dataset.createDimension("pressure", len(pressures))
    dataset.createDimension("time", len(times))
    var = dataset.createVariable("time", "f", ("time",))
    var.axis = "T"
    var.units = units
    var[:] = netCDF4.date2num(times, units=units)
    var = dataset.createVariable("pressure", "f", ("pressure",))
    var.axis = "Z"
    var[:] = pressures
    var = dataset.createVariable("latitude", "f", ("latitude",))
    var.axis = "Y"
    var.long_name = "latitude"
    var[:] = lons
    var = dataset.createVariable("longitude", "f", ("longitude",))
    var.axis = "X"
    var.long_name = "longitude"
    var[:] = lats
    var = dataset.createVariable("air_temperature", "f",
            ("time", "pressure", "latitude", "longitude"))
    var.units = "C"
    var[:] = 30.


def test_dbloader_image(tmpdir):
    path = str(tmpdir / "file.nc")
    times = [dt.datetime(2019, 1, 1)]
    pressures = [1000.]
    lons = [0, 10, 20]
    lats = [0, 10, 20]
    with netCDF4.Dataset(path, "w") as dataset:
        _um_file(dataset, times, pressures, lats, lons)
    locator = unified_model.Locator([path])
    loader = data.DBLoader("Label", "*.nc", locator)
    state = dict(
        variable="air_temperature",
        initial_time=times[0],
        valid_time=times[0],
        pressure=pressures[0],
        pressures=pressures)
    result = loader.image(state)
    assert set(result.keys()) == set([
        "x", "y", "dw", "dh", "image",
        "name", "units", "initial", "valid", "length", "level"])
    assert int(result["x"][0]) == 0
    assert int(result["y"][0]) == 0
    assert int(result["dw"][0]) == 2226389
    assert int(result["dh"][0]) == 2273030
    assert np.all(result["image"][0] == 30.)
    assert result["name"] == ["Label"]
    assert result["units"] == ["C"]
    assert result["valid"] == times
    assert result["initial"] == times
    assert result["length"] == ["T+0"]
    assert result["level"] == ["1000 hPa"]


def test_unified_model_locator(tmpdir):
    path = str(tmpdir / "file.nc")
    times = [dt.datetime(2019, 1, 1)]
    pressures = [1000.]
    lons = [0, 10, 20]
    lats = [0, 10, 20]
    with netCDF4.Dataset(path, "w") as dataset:
        _um_file(dataset, times, pressures, lats, lons)
    locator = unified_model.Locator([path])
    result_path, result_pts = locator.locate(
            "*.nc",
            "air_temperature",
            times[0],
            times[0],
            pressures[0])
    assert result_path == path
    assert result_pts == (0, 0)


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
        loader = data.DBLoader(name, pattern, locator)
        result = loader.image({})
        expect = self.empty_image
        self.assert_dict_equal(expect, result)

    def test_image_given_non_existent_entry_in_database(self):
        name = None
        pattern = None
        database = db.Database.connect(":memory:")
        locator = db.Locator(database.connection)
        state = dict(
            variable="variable",
            initial_time="2019-01-01 00:00:00",
            valid_time="2019-01-01 00:00:00",
            pressure=1000.)
        loader = data.DBLoader(name, pattern, locator)
        result = loader.image(state)
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
        state = dict(
            variable=variable,
            initial_time=initial_time,
            valid_time=valid_time,
            pressure=pressure,
            pressures=[925.])
        loader = data.DBLoader(None, "*.nc", locator)
        result = loader.image(state)
        expect = self.empty_image
        self.assert_dict_equal(expect, result)

    def assert_dict_equal(self, expect, result):
        self.assertEqual(set(expect.keys()), set(result.keys()))
        for key in expect.keys():
            msg = "values not equal for key='{}'".format(key)
            self.assertEqual(expect[key], result[key], msg)
