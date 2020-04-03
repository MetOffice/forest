import pytest
from unittest.mock import Mock
import cftime
import datetime as dt
import bokeh.models
import netCDF4
import numpy as np
import forest.drivers
from forest.drivers import eida50
from forest.exceptions import FileNotFound, IndexNotFound


# Sample data similar to a typical EIDA50 file
TIMES = [dt.datetime(2019, 4, 17) + i * dt.timedelta(minutes=15)
        for i in range(94)]
LONS = np.linspace(-19, 53, 180)  # 10 times fewer for speed
LATS = np.linspace(-13, 23, 90) # 10 times fewer for speed


def _eida50(dataset, times, lons=[0], lats=[0]):
    """Helper to define EIDA50 formatted file"""
    dataset.createDimension("time", len(times))
    dataset.createDimension("longitude", len(lons))
    dataset.createDimension("latitude", len(lats))
    units = "hours since 1970-01-01 00:00:00"
    var = dataset.createVariable(
            "time", "d", ("time",))
    var.axis = "T"
    var.units = units
    var.standard_name = "time"
    var.calendar = "gregorian"
    var[:] = netCDF4.date2num(times, units=units)
    var = dataset.createVariable(
            "longitude", "f", ("longitude",))
    var.axis = "X"
    var.units = "degrees_east"
    var.standard_name = "longitude"
    var[:] = lons
    var = dataset.createVariable(
            "latitude", "f", ("latitude",))
    var.axis = "Y"
    var.units = "degrees_north"
    var.standard_name = "latitude"
    var[:] = lats
    var = dataset.createVariable(
            "data", "f", ("time", "latitude", "longitude"))
    var.standard_name = "toa_brightness_temperature"
    var.long_name = "toa_brightness_temperature"
    var.units = "K"
    var[:] = 0


def test_dataset_navigator():
    settings = {
        "pattern": ""
    }
    dataset = forest.drivers.get_dataset("eida50", settings)
    navigator = dataset.navigator()
    assert navigator.variables(None) == ["EIDA50"]


def test_dataset_map_view():
    settings = {
        "pattern": "",
    }
    color_mapper = bokeh.models.ColorMapper()
    dataset = forest.drivers.get_dataset("eida50", settings)
    view = dataset.map_view(color_mapper)
    view.render({})


def test_navigator_pressures():
    navigator = eida50.Navigator(None, None)
    assert navigator.pressures(None, None, None) == []


@pytest.mark.parametrize("path,expect", [
    pytest.param("/some/file-20190101.nc",
                 dt.datetime(2019, 1, 1),
                 id="yyyymmdd format"),
    pytest.param("/some/file-20190101T1245Z.nc",
                 dt.datetime(2019, 1, 1, 12, 45),
                 id="yyyymmdd hm format"),
    pytest.param("eida50.nc",
                 None,
                 id="no timestamp"),
])
def test_locator_parse_date(path, expect):
    result = eida50.Locator.parse_date(path)
    assert expect == result


def test_locator_find_given_no_files_raises_notfound(tmpdir):
    any_date = dt.datetime.now()
    pattern = str(tmpdir / "nofile.nc")
    locator = eida50.Locator(pattern, eida50.Database())
    with pytest.raises(FileNotFound):
        locator.find([], any_date)


def test_locator_find_given_a_single_file(tmpdir):
    valid_date = dt.datetime(2019, 1, 1)
    path = str(tmpdir / "test-eida50-20190101.nc")
    pattern = str(tmpdir / "test-eida50*.nc")

    times = [valid_date]
    with netCDF4.Dataset(path, "w") as dataset:
        set_times(dataset, times)

    locator = eida50.Locator(pattern, eida50.Database())
    paths = locator.glob()
    found_path, index = locator.find(paths, valid_date)
    assert found_path == path
    assert index == 0


def test_find_given_multiple_files(tmpdir):
    pattern = str(tmpdir / "test-eida50*.nc")
    dates = [
            dt.datetime(2019, 1, 1),
            dt.datetime(2019, 1, 2),
            dt.datetime(2019, 1, 3)]
    for date in dates:
        path = str(tmpdir / "test-eida50-{:%Y%m%d}.nc".format(date))
        with netCDF4.Dataset(path, "w") as dataset:
            set_times(dataset, [date])
    valid_date = dt.datetime(2019, 1, 2, 0, 14)
    locator = eida50.Locator(pattern, eida50.Database())
    paths = locator.glob()
    found_path, index = locator.find(paths, valid_date)
    expect = str(tmpdir / "test-eida50-20190102.nc")
    assert found_path == expect
    assert index == 0


def set_times(dataset, times):
    units = "seconds since 1970-01-01 00:00:00"
    dataset.createDimension("time", len(times))
    var = dataset.createVariable("time", "d", ("time",))
    var.units = units
    var[:] = netCDF4.date2num(times, units=units)


def test_locator_find_index_given_valid_time():
    time = dt.datetime(2019, 1, 1, 3, 31)
    times = [
        dt.datetime(2019, 1, 1, 3, 0),
        dt.datetime(2019, 1, 1, 3, 15),
        dt.datetime(2019, 1, 1, 3, 30),
        dt.datetime(2019, 1, 1, 3, 45),
        dt.datetime(2019, 1, 1, 4, 0),
    ]
    freq = dt.timedelta(minutes=15)
    result = eida50.Locator.find_index(times, time, freq)
    expect = 2
    assert expect == result


def test_locator_find_index_outside_range_raises_exception():
    time = dt.datetime(2019, 1, 4, 16)
    times = [
        dt.datetime(2019, 1, 1, 3, 0),
        dt.datetime(2019, 1, 1, 3, 15),
        dt.datetime(2019, 1, 1, 3, 30),
        dt.datetime(2019, 1, 1, 3, 45),
        dt.datetime(2019, 1, 1, 4, 0),
    ]
    freq = dt.timedelta(minutes=15)
    with pytest.raises(IndexNotFound):
        eida50.Locator.find_index(times, time, freq)


def test_loader_image(tmpdir):
    path = str(tmpdir / "file_20190417.nc")
    with netCDF4.Dataset(path, "w") as dataset:
        _eida50(dataset, TIMES, LONS, LATS)
    time = dt.datetime(2019, 4, 17, 12)
    loader = eida50.Loader(eida50.Locator(path, eida50.Database()))
    image = loader._image(time)
    result = set(image.keys())
    expect = set(["x", "y", "dw", "dh", "image"])
    assert expect == result


def test_loader_longitudes(tmpdir):
    path = str(tmpdir / "eida50_20190417.nc")
    with netCDF4.Dataset(path, "w") as dataset:
        _eida50(dataset, TIMES, LONS, LATS)
    loader = eida50.Loader(eida50.Locator(path, eida50.Database()))
    result = loader.longitudes
    with netCDF4.Dataset(path) as dataset:
        expect = dataset.variables["longitude"][:]
    np.testing.assert_array_almost_equal(expect, result)

def test_loader_latitudes(tmpdir):
    path = str(tmpdir / "eida50_20190417.nc")
    with netCDF4.Dataset(path, "w") as dataset:
        _eida50(dataset, TIMES, LONS, LATS)
    loader = eida50.Loader(eida50.Locator(path, eida50.Database()))
    result = loader.latitudes
    with netCDF4.Dataset(path) as dataset:
        expect = dataset.variables["latitude"][:]
    np.testing.assert_array_almost_equal(expect, result)


def test_locator_times(tmpdir):
    path = str(tmpdir / "eida50_20190417.nc")
    with netCDF4.Dataset(path, "w") as dataset:
        _eida50(dataset, TIMES, LONS, LATS)
    result = eida50.Locator.load_time_axis(path)
    with netCDF4.Dataset(path) as dataset:
        var = dataset.variables["time"]
        expect = netCDF4.num2date(var[:], units=var.units)
    np.testing.assert_array_equal(expect, result)


def test_navigator_initial_times(tmpdir):
    path = str(tmpdir / "test-navigate-eida50.nc")
    settings = {"pattern": path}
    variable = None
    dataset = forest.drivers.get_dataset("eida50", settings)
    navigator = dataset.navigator()
    with netCDF4.Dataset(path, "w") as dataset:
        _eida50(dataset, TIMES)
    result = navigator.initial_times(path, variable)
    expect = [dt.datetime(1970, 1, 1)]
    assert expect == result


def test_navigator_valid_times(tmpdir):
    path = str(tmpdir / "test-navigate-eida50.nc")
    settings = {"pattern": path}
    dataset = forest.drivers.get_dataset("eida50", settings)
    navigator = dataset.navigator()
    with netCDF4.Dataset(path, "w") as dataset:
        _eida50(dataset, TIMES)
    variable = "toa_brightness_temperature"
    result = navigator.valid_times(path, variable, TIMES[0])
    expect = TIMES
    np.testing.assert_array_equal(expect, result)


def test_navigator_parse_times_from_file_names():
    paths = ["eida50_20200101.nc"]
    dataset = forest.drivers.get_dataset("eida50")
    navigator = dataset.navigator()
    result = [navigator.locator.parse_date(path) for path in paths]
    assert result == [dt.datetime(2020, 1, 1)]


def test_database():
    """Basic API for storing/retrieving meta-data"""
    path = "file.nc"
    times = [dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 2)]
    database = forest.drivers.eida50.Database()
    database.insert_times(times[:1], path)
    database.insert_times(times[1:], path)
    assert database.fetch_times() == times
    assert database.fetch_paths() == [path]


def test_fetch_times_given_path():
    paths = ["file-a.nc", "file-b.nc"]
    times = [
        [dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 2)],
        [dt.datetime(2020, 1, 3), dt.datetime(2020, 1, 4)],
    ]
    database = forest.drivers.eida50.Database()
    database.insert_times(times[0], paths[0])
    database.insert_times(times[1], paths[1])
    assert database.fetch_times(paths[0]) == times[0]
    assert database.fetch_times(paths[1]) == times[1]


def test_database_open_twice(tmpdir):
    """sqlite3.OperationalError if table not robust to redefinition"""
    path = str(tmpdir / "file.db")
    with forest.drivers.eida50.Database(path):
        pass
    with forest.drivers.eida50.Database(path):
        pass


@pytest.mark.parametrize("value", [
    None,
    "2020-01-01 00:00:00",
    np.datetime64("2020-01-01 00:00:00", "s")
])
def test_database_insert_times_invalid_types(value):
    """Anything that doesn't support object.strftime(fmt)"""
    database = forest.drivers.eida50.Database()
    with pytest.raises(Exception):
        database.insert_times([value], "file.nc")


@pytest.mark.parametrize("value,expect", [
    pytest.param(dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 1)),
    pytest.param(cftime.DatetimeGregorian(2020, 1, 1), dt.datetime(2020, 1, 1))
])
def test_database_insert_times_supported_types(value, expect):
    database = forest.drivers.eida50.Database()
    database.insert_times([value], "file.nc")
    assert database.fetch_times() == [expect]
