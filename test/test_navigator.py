import pytest
import forest
import datetime as dt
import numpy as np


@pytest.mark.skip("use real unified model file")
def test_unified_model_navigator():
    paths = ["unified.nc"]
    navigator = forest.navigate.FileSystem(
            paths,
            coordinates=forest.unified_model.Coordinates())
    result = navigator.initial_times("*.nc")
    expect = []
    assert expect == result


@pytest.fixture
def rdt_navigator():
    paths = ["/some/file_201901010000.json"]
    return forest.navigate.FileSystem.file_type(paths, "rdt")


def test_rdt_navigator_valid_times_given_single_file(rdt_navigator):
    paths = ["/some/rdt_201901010000.json"]
    navigator = forest.navigate.FileSystem.file_type(paths, "rdt")
    actual = navigator.valid_times("*.json", None, None)
    expected = ["2019-01-01 00:00:00"]
    assert actual == expected


def test_rdt_navigator_valid_times_given_multiple_files(rdt_navigator):
    paths = [
            "/some/rdt_201901011200.json",
            "/some/rdt_201901011215.json",
            "/some/rdt_201901011230.json"
    ]
    navigator = forest.navigate.FileSystem.file_type(paths, "rdt")
    actual = navigator.valid_times(paths[1], None, None)
    expected = ["2019-01-01 12:15:00"]
    np.testing.assert_array_equal(actual, expected)


def test_rdt_navigator_variables(rdt_navigator):
    assert rdt_navigator.variables("*.json") == ["RDT"]


def test_rdt_navigator_initial_times(rdt_navigator):
    assert rdt_navigator.initial_times("*.json") == ["2019-01-01 00:00:00"]


def test_rdt_navigator_pressures(rdt_navigator):
    assert rdt_navigator.pressures("*.json", None, None) == []
