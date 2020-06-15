import datetime as dt
import pytest
import bokeh.models
import forest.drivers
from forest.drivers import saf


@pytest.fixture
def dataset():
    return forest.drivers.get_dataset("saf", {
        "pattern": "saf.nc",
    })


@pytest.fixture
def navigator():
    return saf.Navigator(saf.Locator("fake.nc"))


def test_dataset_map_view(dataset):
    color_mapper = bokeh.models.ColorMapper()
    assert isinstance(dataset.map_view(color_mapper), forest.map_view.ImageView)


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


@pytest.mark.parametrize("paths,date,expect", [
    pytest.param(
        ["S_NWC_CTTH_MSG4_GuineaCoast-VISIR_20191021T134500Z.nc"],
        dt.datetime(2019, 10, 21, 13, 45),
        ["S_NWC_CTTH_MSG4_GuineaCoast-VISIR_20191021T134500Z.nc"],
        id="File name matches date"
    ),
    pytest.param(
        ["S_NWC_CTTH_MSG4_GuineaCoast-VISIR_20191021T134500Z.nc"],
        dt.datetime(2019, 10, 21, 14, 0),
        [],
        id="File name earlier than date"
    ),
    pytest.param(
        ["S_NWC_CTTH_MSG4_GuineaCoast-VISIR_20191021T134500Z.nc"],
        dt.datetime(2019, 10, 21, 13, 55),
        ["S_NWC_CTTH_MSG4_GuineaCoast-VISIR_20191021T134500Z.nc"],
        id="Time inside window"
    ),
])
def test_file_name_locator(paths, date, expect):
    pattern = ""
    locator = saf.Locator(pattern, )
    frequency = dt.timedelta(minutes=15)
    assert list(locator.find_paths(paths, date, frequency)) == expect
