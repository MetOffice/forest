from unittest.mock import Mock, call, patch, sentinel

import numpy as np
import pytest

import forest.exceptions

import forest.navigate as navigate


def test_get_database_with_real_dependencies(tmpdir):
    # Note: needed to check os library imported
    database_path = str(tmpdir / "example.db")
    with pytest.raises(ValueError):
        forest.db.get_database(database_path)


def test_Navigator_variables():
    sub_navigator = Mock()
    sub_navigator.variables.return_value = sentinel.variables
    navigator = Mock(_navigators={sentinel.pattern: sub_navigator})

    result = navigate.Navigator.variables(navigator, sentinel.pattern)

    sub_navigator.variables.assert_called_once_with(sentinel.pattern)
    assert result == sentinel.variables


def test_Navigator_initial_times():
    sub_navigator = Mock()
    sub_navigator.initial_times.return_value = sentinel.initial_times
    navigator = Mock(_navigators={sentinel.pattern: sub_navigator})

    result = navigate.Navigator.initial_times(navigator, sentinel.pattern,
                                              sentinel.variable)

    sub_navigator.initial_times.assert_called_once_with(
        sentinel.pattern, variable=sentinel.variable)
    assert result == sentinel.initial_times


def test_Navigator_valid_times():
    sub_navigator = Mock()
    sub_navigator.valid_times.return_value = sentinel.valid_times
    navigator = Mock(_navigators={sentinel.pattern: sub_navigator})

    result = navigate.Navigator.valid_times(navigator, sentinel.pattern,
                                            sentinel.variable,
                                            sentinel.initial_time)

    sub_navigator.valid_times.assert_called_once_with(
        sentinel.pattern, sentinel.variable, sentinel.initial_time)
    assert result == sentinel.valid_times


def test_Navigator_pressures():
    sub_navigator = Mock()
    sub_navigator.pressures.return_value = sentinel.pressures
    navigator = Mock(_navigators={sentinel.pattern: sub_navigator})

    result = navigate.Navigator.pressures(navigator, sentinel.pattern,
                                          sentinel.variable,
                                          sentinel.initial_time)

    sub_navigator.pressures.assert_called_once_with(
        sentinel.pattern, sentinel.variable, sentinel.initial_time)
    assert result == sentinel.pressures


@patch('forest.drivers.gridded_forecast.Navigator')
@patch('forest.drivers.gridded_forecast.glob.glob')
def test_drivers__get_dataset_from__griddedforecast(glob, navigator_cls):
    navigator_cls.return_value = sentinel.navigator
    glob.return_value = sentinel.paths

    dataset = forest.drivers.gridded_forecast.Dataset("gridded_forecast", sentinel.settings)
    navigator = dataset.navigator()

    navigator_cls.assert_called_once_with(sentinel.paths)
    assert navigator == sentinel.navigator
