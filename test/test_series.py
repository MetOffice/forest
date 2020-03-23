import pytest
import unittest
import os
import netCDF4
import cftime
import numpy as np
import numpy.testing as npt
import datetime as dt
import bokeh.plotting
from forest import screen, series, redux, rx, config


@pytest.mark.parametrize("state,expect", [
        ({}, None),
        ({"position": {"x": 0, "y": 0}}, None),
        ({"variable": "mslp", "position": {"x": 1, "y": 2}}, None),
        ({
            "variable": "mslp",
            "initial_time": "2019-01-01 00:00:00",
            "position": {"x": 1, "y": 2},
            "tools": {"time_series": True}},
            (dt.datetime(2019, 1, 1), "mslp", 1, 2, True)),
        ({
            "variable": "air_temperature",
            "pressure": 1000.,
            "initial_time": "2019-01-01 00:00:00",
            "position": {"x": 1, "y": 2},
            "tools": {"time_series": False}},
            (dt.datetime(2019, 1, 1), "air_temperature", 1, 2, False, 1000.)),
    ])
def test_select_args(state, expect):
    result = series.select_args(state)
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


def test_series_view():
    figure = bokeh.plotting.figure()
    series.SeriesView(figure, {})


def test_series_view_render():
    figure = bokeh.plotting.figure()
    view = series.SeriesView(figure, {})
    time = dt.datetime.now()
    variable = "mslp"
    x = 0
    y = 0
    visible = True
    view.render(time, variable, x, y, visible)


def test_series_view_from_groups():
    figure = bokeh.plotting.figure()
    group = config.FileGroup("label", "pattern")
    series.SeriesView.from_groups(figure, [group])


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
            "relative_humidity", "f",
            ("dim0", "latitude", "longitude"))
    var.units = "%"
    var.grid_mapping = "latitude_longitude"
    var.coordinates = "forecast_period forecast_reference_time pressure time"
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
            variable, "f",
            ("time", "latitude", "longitude"))
    var.units = "Pa"
    var.grid_mapping = "latitude_longitude"
    var.coordinates = "forecast_period forecast_reference_time"
    var[:] = values


def variable_3d_scalar_time(
        dataset,
        variable,
        time,
        pressures,
        longitudes,
        latitudes,
        values):
    dataset.createDimension("latitude", len(latitudes))
    dataset.createDimension("longitude", len(longitudes))
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
    units = "hours since 1970-01-01 00:00:00"
    var = dataset.createVariable(
            "pressure", "d", ("pressure",))
    var[:] = pressures
    var = dataset.createVariable(
            "time", "d", ())
    var.units = units
    var[:] = netCDF4.date2num(time, units=units)
    var = dataset.createVariable(
            variable, "f",
            ("pressure", "latitude", "longitude"))
    var.units = "%"
    var.grid_mapping = "latitude_longitude"
    var.coordinates = "forecast_period forecast_reference_time time"
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
    dataset.createDimension("time_1", len(times))
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
    units = "hours since 1970-01-01 00:00:00"
    var = dataset.createVariable(
            "pressure", "d", ("pressure",))
    var[:] = pressures
    var = dataset.createVariable(
            "time_1", "d", ("time_1",))
    var.units = units
    var[:] = netCDF4.date2num(times, units=units)
    var = dataset.createVariable(
            variable, "f",
            ("time_1", "pressure", "latitude", "longitude"))
    var.units = "K"
    var.grid_mapping = "latitude_longitude"
    var.coordinates = "forecast_period_1 forecast_reference_time"
    var[:] = values


def test_3d_variable_scalar_time(tmpdir):
    path = str(tmpdir / "file.nc")
    variable = "relative_humidity"
    time = dt.datetime(2019, 1, 1)
    pressures = [
            1000.001,
            500,
            250]
    longitudes = [0, 1]
    latitudes = [0, 1]
    values = np.arange(3*2*2).reshape(3, 2, 2)
    with netCDF4.Dataset(path, "w") as dataset:
        variable_3d_scalar_time(
                dataset,
                variable,
                time,
                pressures,
                longitudes,
                latitudes,
                values)
    lon, lat = 0.1, 0.1
    loader = series.SeriesLoader([path])
    result = loader._load_netcdf4(
            path,
            variable,
            lon,
            lat,
            pressure=500)
    expect = {
        "x": [time],
        "y": [values[1, 0, 0]]
    }
    npt.assert_array_equal(expect["x"], result["x"])
    npt.assert_array_equal(expect["y"], result["y"])


def test_4d_variable(tmpdir):
    path = str(tmpdir / "file.nc")
    variable = "wet_bulb_potential_temperature"
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
    with netCDF4.Dataset(path, "w") as dataset:
        variable_4d(
                dataset,
                variable,
                times,
                pressures,
                longitudes,
                latitudes,
                values)
    lon, lat = 0.1, 0.1
    loader = series.SeriesLoader([path])
    result = loader._load_netcdf4(
            path,
            variable,
            lon,
            lat,
            pressure=500)
    expect = {
        "x": times,
        "y": values[:, 1, 0, 0]
    }
    npt.assert_array_equal(expect["x"], result["x"])
    npt.assert_array_equal(expect["y"], result["y"])


@pytest.mark.parametrize("value,expect", [
    (dt.datetime(2020, 1, 1), "2020-01-01 00:00:00"),
    (cftime.DatetimeGregorian(2020, 1, 1), "2020-01-01 00:00:00")
])
def test_series_locator_key(value, expect):
    assert series.SeriesLocator.key(value) == expect


class TestSeries(unittest.TestCase):
    def setUp(self):
        self.path = "test-series.nc"

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_series_given_missing_variable_returns_empty(self):
        pressure = 500
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
        loader = series.SeriesLoader([self.path])
        variable = "not_in_file"
        result = loader.series_file(
                self.path, variable, lon, lat, pressure)
        expect = {
            "x": [],
            "y": []
        }
        npt.assert_array_equal(expect["x"], result["x"])
        npt.assert_array_equal(expect["y"], result["y"])

    def test_series_given_dim0_variable(self):
        variable = "relative_humidity"
        pressure = 500
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
        loader = series.SeriesLoader([self.path])
        result = loader.series_file(
                self.path, variable, lon, lat, pressure)
        i, j = 1, 1
        expect = {
            "x": [t0, t1],
            "y": [values[1, j, i], values[3, j, i]]
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
        loader = series.SeriesLoader([self.path])
        result = loader.series_file(
                self.path, variable, lon, lat)
        expect = {
            "x": times,
            "y": values[:, 1, 0]
        }
        npt.assert_array_equal(expect["x"], result["x"])
        npt.assert_array_equal(expect["y"], result["y"])

    def test_series_locator(self):
        paths = [
            "/some/file_20190101T0000Z_000.nc",
            "/some/file_20190101T0000Z_006.nc",
            "/some/file_20190101T0000Z_012.nc",
            "/some/file_20190101T1200Z_000.nc",
            "/some/file_20190101T1200Z_006.nc",
            "/some/file_20190101T1200Z_012.nc",
        ]
        reference_time = dt.datetime(2019, 1, 1, 12)
        locator = series.SeriesLocator(paths)
        result = locator[reference_time]
        expect = [
            "/some/file_20190101T1200Z_000.nc",
            "/some/file_20190101T1200Z_006.nc",
            "/some/file_20190101T1200Z_012.nc",
        ]
        self.assertEqual(expect, result)

    def test_series_locator_getitem_given_datetime64(self):
        paths = [
            "/some/file_20190101T0000Z_000.nc",
            "/some/file_20190101T0000Z_006.nc",
            "/some/file_20190101T0000Z_012.nc",
            "/some/file_20190101T1200Z_000.nc",
            "/some/file_20190101T1200Z_006.nc",
            "/some/file_20190101T1200Z_012.nc",
        ]
        reference_time = np.datetime64('2019-01-01T12:00:00', 's')
        locator = series.SeriesLocator(paths)
        result = locator[reference_time]
        expect = [
            "/some/file_20190101T1200Z_000.nc",
            "/some/file_20190101T1200Z_006.nc",
            "/some/file_20190101T1200Z_012.nc",
        ]
        self.assertEqual(expect, result)

    def test_series_locator_initial_times(self):
        paths = [
            "/some/file_20190101T0000Z_000.nc",
            "/some/file_20190101T0000Z_006.nc",
            "/some/file_20190101T0000Z_012.nc",
            "/some/file_20190101T1200Z_000.nc",
            "/some/file_20190101T1200Z_006.nc",
            "/some/file_20190101T1200Z_012.nc",
        ]
        locator = series.SeriesLocator(paths)
        result = locator.initial_times()
        expect = np.array([
            '2019-01-01 00:00',
            '2019-01-01 12:00'], dtype='datetime64[s]')
        npt.assert_array_equal(expect, result)

    def test_pressures_matches_large_pressures(self):
        pressures = np.array([1000.001, 1000.01, 1000.1, 950])
        result = series.SeriesLoader.search(pressures, 1000)
        expect = np.array([True, True, True, False])
        npt.assert_array_equal(expect, result)

    def test_pressures_matches_small_pressures(self):
        pressures = np.array([0.03001, 0.020001, 0.010001])
        result = series.SeriesLoader.search(pressures, 0.02)
        expect = np.array([False, True, False])
        npt.assert_array_equal(expect, result)
