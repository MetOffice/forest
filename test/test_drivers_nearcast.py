import datetime as dt
import pytest
import unittest.mock
from unittest.mock import Mock, sentinel, patch
import bokeh.models
import pandas as pd
import pandas.testing as pdt
import pygrib
import forest.drivers
import forest.map_view
from forest.drivers import nearcast


def test_dataset_navigator():
    dataset = forest.drivers.get_dataset("nearcast")
    navigator = dataset.navigator()
    assert isinstance(navigator, forest.drivers.nearcast.Navigator)


def test_dataset_map_view():
    color_mapper = bokeh.models.ColorMapper()
    dataset = forest.drivers.get_dataset("nearcast")
    map_view = dataset.map_view(color_mapper)
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


def test_navigator_initial_times_given_large_number_of_files():
    """should not open file(s)"""
    pattern = ""
    variable = None
    times = pd.date_range("2020-01-01 00:00:00",
                          periods=1000,
                          freq="30min")
    paths = [f"NEARCAST_{time:%Y%m%d_%H%M}_LAKEVIC_LATLON.GRIB2"
             for time in times]
    with unittest.mock.patch("forest.drivers.nearcast.glob") as glob:
        glob.glob.return_value = paths
        navigator = nearcast.Navigator(pattern)
        result = pd.to_datetime(navigator.initial_times(pattern, variable))
        expect = times
        pdt.assert_index_equal(expect, result)


@patch("forest.drivers.nearcast.pg")
@patch("forest.drivers.nearcast.glob")
def test_navigator_valid_times_given_large_number_of_files(glob, pygrib):
    """should only open one file to find valid dates"""
    pattern = "pattern"
    initial_time = "2020-01-01 03:30:00"
    times = pd.date_range("2020-01-01 00:00:00",
                          periods=1000,
                          freq="30min")
    paths = [f"NEARCAST_{time:%Y%m%d_%H%M}_LAKEVIC_LATLON.GRIB2"
             for time in times]

    glob.glob.return_value = paths
    messages = Mock()
    messages.select.return_value = [{
        "validityDate": 20200101,
        "validityTime": 330
    }]
    pygrib.index.return_value = messages

    navigator = nearcast.Navigator(sentinel.pattern)
    result = navigator.valid_times(sentinel.pattern_not_used,
                                   sentinel.variable,
                                   initial_time)

    glob.glob.assert_called_once_with(sentinel.pattern)
    pygrib.index.assert_called_once_with(
        "NEARCAST_20200101_0330_LAKEVIC_LATLON.GRIB2", "name")
    messages.select.assert_called_once_with(name=sentinel.variable)
    assert result == [dt.datetime(2020, 1, 1, 3, 30)]
