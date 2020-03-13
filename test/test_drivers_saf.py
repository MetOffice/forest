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
    return saf.Navigator()


def test_dataset_map_view(dataset):
    assert isinstance(dataset.map_view(), forest.view.UMView)


def test_dataset_navigator(dataset):
    assert isinstance(dataset.navigator(), saf.Navigator)


def test_navigator_variables(navigator):
    pattern = "saf.nc"
    assert navigator.variables(pattern) == []


def test_navigator_initial_times(navigator):
    pattern, variable = "saf.nc", None
    assert navigator.initial_times(pattern, variable) == []


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
    loader = saf.saf("saf.nc")
    loader._image(variable, initial_time, valid_time, pressures, pressure)
