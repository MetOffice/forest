import datetime as dt
import pytest
import bokeh.models
import forest.drivers
from forest.drivers import saf


@pytest.fixture
def dataset():
    color_mapper = bokeh.models.ColorMapper()
    return forest.drivers.get_dataset("saf", {
        "pattern": "saf.nc",
        "color_mapper": color_mapper
    })


@pytest.fixture
def navigator():
    return saf.Navigator(saf.Locator("fake.nc"))


def test_dataset_map_view(dataset):
    assert isinstance(dataset.map_view(), forest.view.UMView)


def test_dataset_navigator(dataset):
    assert isinstance(dataset.navigator(), saf.Navigator)


def test_navigator_variables(navigator):
    pattern = "saf.nc"
    assert navigator.variables(pattern) == []


def test_navigator_initial_times(navigator):
    pattern, variable = "saf.nc", None
    assert navigator.initial_times(pattern, variable) == [dt.datetime(1970, 1, 1)]


def test_navigator_valid_times(navigator):
    pattern, variable, initial_time = "saf.nc", None, None
    assert navigator.valid_times(pattern, variable, initial_time) == []


def test_navigator_pressures(navigator):
    pattern, variable, initial_time = "saf.nc", None, None
    assert navigator.pressures(pattern, variable, initial_time) == []


def test_loader():
    # Create seam to pass test data
    variable = None
    initial_time = None
    valid_time = None
    pressures = None
    pressure = None
    loader = saf.Loader(saf.Locator("saf.nc"))
    loader._image(variable, initial_time, valid_time, pressures, pressure)


@pytest.mark.parametrize("regex,fmt,paths,date,expect", [
    pytest.param(
        "[0-9]{8}", "%Y%m%d", ["some_20200101.nc"],
        dt.datetime(2020, 1, 1), ["some_20200101.nc"]
    ),
    pytest.param(
        "[0-9]{8}", "%Y%m%d", ["some_20200101.nc"],
        dt.datetime(2020, 1, 2), []
    ),
])
def test_file_name_locator(regex, fmt, paths, date, expect):
    locator = saf.FileNameLocator(regex, fmt)
    assert list(locator.find_paths(paths, date)) == expect


@pytest.mark.parametrize("regex,fmt,path,expect", [
    pytest.param(
        "[0-9]{8}", "%Y%m%d", "some_20200101.nc", dt.datetime(2020, 1, 1)
    ),
    pytest.param(
        "[0-9]{8}", "%Y%m%d", "file.nc", None,
        id="No match"
    ),
    pytest.param(
        "[0-9]{8}T[0-9]{6}Z", "%Y%m%dT%H%M%S%Z",
        "S_NWC_CTTH_MSG4_GuineaCoast-VISIR_20191021T134500Z.nc",
        dt.datetime(2019, 10, 21, 13, 45),
        id="SAF format"
    ),
])
def test_file_name_locator_parse_date(regex, fmt, path, expect):
    locator = saf.FileNameLocator(regex, fmt)
    assert locator.parse_date(path) == expect
