import pytest
import unittest
import os
import netCDF4
import numpy as np
import numpy.testing as npt
import datetime as dt
import bokeh.plotting
from forest import screen, redux, rx, config
from forest import _profile as profile


@pytest.mark.parametrize("state,expect", [
        ({}, None),
        ({"position": {"x": 0, "y": 0}}, None),
        ({"variable": "mslp", "position": {"x": 1, "y": 2}}, None),
        ({
            "variable": "mslp",
            "initial_time": "2019-01-01 00:00:00",
            "position": {"x": 1, "y": 2},
            "tools": {"profile": True}},
            (dt.datetime(2019, 1, 1), "mslp", 1, 2, True)),
        ({
            "variable": "air_temperature",
            "pressure": 1000.,
            "initial_time": "2019-01-01 00:00:00",
            "position": {"x": 1, "y": 2},
            "tools": {"profile": False},
            "valid_time": "2019-01-01 00:00:00"},
            (dt.datetime(2019, 1, 1), "air_temperature", 1, 2, False, dt.datetime(2019, 1, 1))),
    ])
def test_select_args(state, expect):
    result = profile.select_args(state)
    assert expect == result


@pytest.mark.parametrize("values,expect", [
        ([], []),
        ([1, 2, 3, 3, 4], [1, 2, 3, 4]),
        ([None, None, None, None], [None]),
        ([{"x": 2}, {"x": 1}, {"x": 1}], [{"x": 2}, {"x": 1}]),
    ])
def test_stream_distinct(values, expect):
    stream = rx.Stream()
    deduped = stream.distinct()
    listener = unittest.mock.Mock()
    deduped.add_subscriber(listener)
    for value in values:
        stream.notify(value)
    calls = [unittest.mock.call(v) for v in expect]
    listener.assert_has_calls(calls)
    assert listener.call_count == len(calls)


@pytest.mark.parametrize("values,predicate,expect", [
    ([], lambda x: True, []),
    ([1, 2, 3], lambda x: True, [1, 2, 3]),
    ([1, 2, 3, 4], lambda x: x > 2, [3, 4]),
    ([1, 2, None, 3, 4], lambda x: x is not None, [1, 2, 3, 4]),
    ])
def test_stream_filter(values, predicate, expect):
    stream = rx.Stream()
    filtered = stream.filter(predicate)
    listener = unittest.mock.Mock()
    filtered.add_subscriber(listener)
    for value in values:
        stream.notify(value)
    calls = [unittest.mock.call(v) for v in expect]
    listener.assert_has_calls(calls)


def test_profile_view():
    figure = bokeh.plotting.figure()
    profile.ProfileView(figure, {})


def test_profile_view_render():
    figure = bokeh.plotting.figure()
    view = profile.ProfileView(figure, {})
    time = dt.datetime.now()
    variable = "mslp"
    x = 0
    y = 0
    visible = True
    view.render(time, variable, x, y, visible)


def test_profile_view_from_groups():
    figure = bokeh.plotting.figure()
    group = config.FileGroup("label", "pattern")
    profile.ProfileView.from_groups(figure, [group])


def variable_dim0(
        dataset,
        pressures,
        times,
        longitudes,
        latitudes,
        values):
    dataset.createDimension("latitude", len(latitudes))
    dataset.createDimension("longitude", len(longitudes))
    dataset.createDimension("dim0", len(pressures))
    var = dataset.createVariable(
            "longitude", "d", ("longitude",))
    var.axis = "X"
    var.units = "degrees_east"
    var.standard_name = "longitude"
    var[:] = longitudes
    var = dataset.createVariable(
            "latitude", "d", ("latitude",))
    var.axis = "Y"
    var.units = "degrees_north"
    var.standard_name = "latitude"
    var[:] = latitudes
    var = dataset.createVariable(
            "pressure", "d", ("dim0",))
    var[:] = pressures
    units = "hours since 1970-01-01 00:00:00"
    var = dataset.createVariable(
            "time", "d", ("dim0",))
    var.units = units
    var[:] = netCDF4.date2num(times, units=units)
    var = dataset.createVariable(
            "forecast_reference_time", "d", ())
    units = "hours since 1970-01-01 00:00:00"
    var.units = units
    var[:] = netCDF4.date2num(times, units=units)[0]
    var = dataset.createVariable(
            "relative_humidity", "f",
            ("dim0", "latitude", "longitude"))
    var.units = "%"
    var.grid_mapping = "latitude_longitude"
    var.coordinates = "forecast_period forecast_reference_time pressure time"
    var[:] = values


def variable_4d(
        dataset,
        variable,
        times,
        pressures,
        longitudes,
        latitudes,
        values):
    dataset.createDimension("latitude", len(latitudes))
    dataset.createDimension("longitude", len(longitudes))
    dataset.createDimension("time", len(times))
    dataset.createDimension("pressure", len(pressures))
    var = dataset.createVariable(
            "longitude", "d", ("longitude",))
    var.axis = "X"
    var.units = "degrees_east"
    var.standard_name = "longitude"
    var[:] = longitudes
    var = dataset.createVariable(
            "latitude", "d", ("latitude",))
    var.axis = "Y"
    var.units = "degrees_north"
    var.standard_name = "latitude"
    var[:] = latitudes
    var = dataset.createVariable(
            "pressure", "d", ("pressure",))
    var[:] = pressures
    var = dataset.createVariable(
            "time", "d", ("time",))
    units = "hours since 1970-01-01 00:00:00"
    var.units = units
    var[:] = netCDF4.date2num(times, units=units)
    var.standard_name = "time"
    var = dataset.createVariable(
            "forecast_reference_time", "d", ())
    units = "hours since 1970-01-01 00:00:00"
    var.units = units
    var[:] = netCDF4.date2num(times, units=units)[0]
    var = dataset.createVariable(
            variable, "f",
            ("time", "pressure", "latitude", "longitude"))
    var.units = "K"
    var.grid_mapping = "latitude_longitude"
    var.coordinates = "forecast_period_1 forecast_reference_time"
    var[:] = values



def variable_surface(
        dataset,
        variable,
        times,
        longitudes,
        latitudes,
        values):
    dataset.createDimension("latitude", len(latitudes))
    dataset.createDimension("longitude", len(longitudes))
    dataset.createDimension("time", len(times))
    var = dataset.createVariable(
            "longitude", "d", ("longitude",))
    var.axis = "X"
    var.units = "degrees_east"
    var.standard_name = "longitude"
    var[:] = longitudes
    var = dataset.createVariable(
            "latitude", "d", ("latitude",))
    var.axis = "Y"
    var.units = "degrees_north"
    var.standard_name = "latitude"
    var[:] = latitudes
    units = "hours since 1970-01-01 00:00:00"
    var = dataset.createVariable(
            "time", "d", ("time",))
    var.units = units
    var[:] = netCDF4.date2num(times, units=units)
    var = dataset.createVariable(
            "forecast_reference_time", "d", ())
    units = "hours since 1970-01-01 00:00:00"
    var.units = units
    var[:] = netCDF4.date2num(times, units=units)[0]
    var = dataset.createVariable(
            variable, "f",
            ("time", "latitude", "longitude"))
    var.units = "Pa"
    var.grid_mapping = "latitude_longitude"
    var.coordinates = "forecast_period forecast_reference_time"
    var[:] = values


class TestProfile(unittest.TestCase):
    def setUp(self):
        self.path = "test-profile.nc"

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_profile_given_missing_variable_returns_empty(self):
        pressure = 500
        lon = 1
        lat = 1
        times = [
            dt.datetime(2019, 1, 1),
            dt.datetime(2019, 1, 1, 6),
            dt.datetime(2019, 1, 1, 12)]
        pressures = [
                1000.001,
                500,
                250]
        longitudes = [0, 1, 2]
        latitudes = [0, 1, 2]
        values = np.arange(3*3*3*3).reshape(3, 3, 3, 3)
        with netCDF4.Dataset(self.path, "w") as dataset:
            variable_4d(
                dataset,
                "var_in_file",
                times,
                pressures,
                longitudes,
                latitudes,
                values)

        loader = profile.ProfileLoader([self.path])
        variable = "var_not_in_file"
        result = loader.profile_file(
                self.path, variable, lon, lat, pressure)
        expect = {
            "x": [],
            "y": []
        }
        npt.assert_array_equal(expect["x"], result["x"])
        npt.assert_array_equal(expect["y"], result["y"])

    def test_profile_given_dim0_variable(self):
        variable = "relative_humidity"
        lon = 1
        lat = 1
        p0, p1 = 1000, 500
        t0 = dt.datetime(2019, 1, 1)
        t1 = dt.datetime(2019, 1, 1, 3)
        longitudes = [0, 1]
        latitudes = [0, 1]
        pressures = [p0, p1, p0, p1]
        times = [t0, t0, t1, t1]
        values = np.arange(4*2*2).reshape(4, 2, 2)
        with netCDF4.Dataset(self.path, "w") as dataset:
            variable_dim0(
                dataset,
                pressures,
                times,
                longitudes,
                latitudes,
                values)
        loader = profile.ProfileLoader([self.path])
        result = loader.profile_file(
                self.path, variable, lon, lat, t0)
        i, j = 1, 1
        expect = {
            "x": [values[0, j, i], values[1, j, i]],
            "y": [p0, p1]
        }
        npt.assert_array_equal(expect["x"], result["x"])
        npt.assert_array_equal(expect["y"], result["y"])

    def test_surface_variable(self):
        variable = "air_pressure_at_sea_level"
        times = [
                dt.datetime(2019, 1, 1),
                dt.datetime(2019, 1, 1, 12)]
        longitudes = [0, 1, 2]
        latitudes = [0, 1, 2]
        values = np.arange(2*3*3).reshape(2, 3, 3)
        with netCDF4.Dataset(self.path, "w") as dataset:
            variable_surface(
                    dataset,
                    variable,
                    times,
                    longitudes,
                    latitudes,
                    values)
        lon = 0
        lat = 1
        loader = profile.ProfileLoader([self.path])
        result = loader.profile_file(
                self.path, variable, lon, lat)
        expect = {
            "x": [values[0, 1, 0], ],
            "y": [0,]
        }
        npt.assert_array_equal(expect["x"], result["x"])
        npt.assert_array_equal(expect["y"], result["y"])

    def test_profile_locator(self):
        paths = [
            "/some/file_20190101T0000Z_000.nc",
            "/some/file_20190101T0000Z_006.nc",
            "/some/file_20190101T0000Z_012.nc",
            "/some/file_20190101T1200Z_000.nc",
            "/some/file_20190101T1200Z_006.nc",
            "/some/file_20190101T1200Z_012.nc",
        ]
        reference_time = dt.datetime(2019, 1, 1, 12)
        locator = profile.ProfileLocator(paths)
        result = locator.locate(reference_time)
        expect = [
            "/some/file_20190101T1200Z_000.nc",
            "/some/file_20190101T1200Z_006.nc",
            "/some/file_20190101T1200Z_012.nc",
        ]
        self.assertEqual(expect, result)

    def test_profile_locator_getitem_given_datetime64(self):
        paths = [
            "/some/file_20190101T0000Z_000.nc",
            "/some/file_20190101T0000Z_006.nc",
            "/some/file_20190101T0000Z_012.nc",
            "/some/file_20190101T1200Z_000.nc",
            "/some/file_20190101T1200Z_006.nc",
            "/some/file_20190101T1200Z_012.nc",
        ]
        reference_time = np.datetime64('2019-01-01T12:00:00', 's')
        locator = profile.ProfileLocator(paths)
        result = locator.locate(reference_time)
        expect = [
            "/some/file_20190101T1200Z_000.nc",
            "/some/file_20190101T1200Z_006.nc",
            "/some/file_20190101T1200Z_012.nc",
        ]
        self.assertEqual(expect, result)

    def test_profile_locator_initial_times(self):
        paths = [
            "/some/file_20190101T0000Z_000.nc",
            "/some/file_20190101T0000Z_006.nc",
            "/some/file_20190101T0000Z_012.nc",
            "/some/file_20190101T1200Z_000.nc",
            "/some/file_20190101T1200Z_006.nc",
            "/some/file_20190101T1200Z_012.nc",
        ]
        locator = profile.ProfileLocator(paths)
        result = locator.initial_times()
        expect = np.array([
            '2019-01-01 00:00',
            '2019-01-01 12:00'], dtype='datetime64[s]')
        npt.assert_array_equal(expect, result)
