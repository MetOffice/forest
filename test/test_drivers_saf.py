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


def test_dataset_map_view(dataset):
    assert isinstance(dataset.map_view(), forest.view.UMView)


def test_dataset_navigator(dataset):
    assert isinstance(dataset.navigator(), saf.Navigator)


def test_navigator_interface():
    navigator = saf.Navigator()
    assert navigator.variables() == []
    assert navigator.initial_times() == []
    assert navigator.valid_times() == []
    assert navigator.pressures() == []
