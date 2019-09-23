import pytest
import forest
import datetime as dt


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
    paths = ["/some/rdt_201901010000.json"]
    return forest.navigate.FileSystem(
            paths,
            coordinates=forest.rdt.Coordinates())


def test_rdt_navigator_valid_times(rdt_navigator):
    assert rdt_navigator.valid_times("*.json", None, None) == [
            "2019-01-01 00:00:00"]


def test_rdt_navigator_variables(rdt_navigator):
    assert rdt_navigator.variables("*.json") == ["RDT"]


def test_rdt_navigator_initial_times(rdt_navigator):
    assert rdt_navigator.initial_times("*.json") == ["2019-01-01 00:00:00"]


def test_rdt_navigator_pressures(rdt_navigator):
    assert rdt_navigator.pressures("*.json", None, None) == []
