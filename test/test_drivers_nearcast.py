import pytest
import unittest.mock
import bokeh.models
import pygrib
import forest.drivers
import forest.view
from forest.drivers import nearcast


def test_dataset_navigator():
    dataset = forest.drivers.get_dataset("nearcast")
    navigator = dataset.navigator()
    assert isinstance(navigator, forest.drivers.nearcast.Navigator)


def test_dataset_map_view():
    color_mapper = bokeh.models.ColorMapper()
    dataset = forest.drivers.get_dataset("nearcast")
    map_view = dataset.map_view(color_mapper)
    assert isinstance(map_view, forest.view.NearCast)
    assert map_view.tooltips == forest.drivers.nearcast.NEARCAST_TOOLTIPS


def make_open(names):
    # Simulate pygrib.open(path).select() -> messages
    def _open(path):
        messages = unittest.mock.Mock()
        messages.select.return_value = [
            {"name": name} for name in names
        ]
        return messages
    return _open


@pytest.mark.parametrize("names,expect", [
    (["A", "A", "A"], ["A"]),
    (["D", "C", "B", "A"], ["A", "B", "C", "D"]),
])
def test_navigator_variables(monkeypatch, names, expect):
    pattern = ""
    navigator = nearcast.Navigator(pattern)
    navigator.locator = unittest.mock.Mock()
    navigator.locator.find.return_value = ["some.grib"]
    monkeypatch.setattr(pygrib, "open", make_open(names))
    assert navigator.variables(pattern) == expect
