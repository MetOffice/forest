import pytest
import unittest
from unittest.mock import patch
import datetime as dt
import os
import glob
import json
import numpy as np
import forest.drivers
from forest.drivers import rdt
from forest import (
        locate)


def test_dataset_navigator_standard_dimensions():
    settings = {}
    dataset = forest.drivers.get_dataset("rdt", settings)
    navigator = dataset.navigator()
    assert navigator.variables() == ["RDT"]
    assert navigator.initial_times() == [dt.datetime(1970, 1, 1)]
    assert navigator.pressures() == []


def test_dataset_navigator_valid_times():
    pattern = "*.json"
    settings = {"pattern": pattern}
    dataset = forest.drivers.get_dataset("rdt", settings)
    navigator = dataset.navigator()
    with patch("forest.drivers.rdt.glob") as glob:
        glob.glob.return_value = ["rdt_202001010000.json"]
        assert navigator.valid_times() == [dt.datetime(2020, 1, 1)]


def test_loader(tmpdir):
    path = str(tmpdir / "rdt_202001010000.json")
    content = """
{
    "features": [
    ]
}
"""
    with open(path, "w") as stream:
        stream.write(content)
    loader = rdt.Loader(path)
    json_texts = loader.load_date(dt.datetime(2020, 1, 1))
    polygons = json_texts[0]
    result = json.loads(polygons)
    assert result == {"features": []}


@pytest.mark.parametrize("state", [
    {},
    {"valid_time": dt.datetime(2020, 1, 1)},
])
def test_dataset_map_view(state):
    settings = {"pattern": ""}
    dataset = forest.drivers.get_dataset("rdt", settings)
    map_view = dataset.map_view()
    map_view.render(state)


class TestLocator(unittest.TestCase):
    def setUp(self):
        pattern = os.path.join(os.path.dirname(__file__),
                "sample/RDT*.json")
        self.locator = rdt.Locator(pattern)

    def test_paths(self):
        result = [os.path.basename(path) for path in self.locator.paths]
        expect = ["RDT_features_eastafrica_201904171245.json"]
        self.assertEqual(expect, result)

    def test_find_file(self):
        date = dt.datetime(2019, 4, 17, 12, 59)
        result = os.path.basename(self.locator.find_file(date))
        expect = "RDT_features_eastafrica_201904171245.json"
        self.assertEqual(expect, result)

    def test_parse_date(self):
        path = "/Users/andrewryan/cache/RDT_features_eastafrica_201903151215.json"
        result = self.locator.parse_date(path)
        expect = dt.datetime(2019, 3, 15, 12, 15)
        self.assertEqual(expect, result)


def test_time_bounds():
    dates = [dt.datetime(2019, 8, 1)]
    length = dt.timedelta(minutes=15)
    result = locate.bounds(dates, length)
    expect = np.array([
        ['2019-08-01 00:00:00', '2019-08-01 00:15:00']],
        dtype='datetime64[s]')
    np.testing.assert_array_equal(expect, result)


def test_in_bounds():
    time = '2019-08-01 00:14:59'
    bounds = locate.bounds(['2019-08-01 00:00:00'], dt.timedelta(minutes=15))
    result = locate.in_bounds(bounds, time)
    expect = [True]
    np.testing.assert_array_equal(expect, result)


def test_in_bounds_given_point_outside_bounds():
    time = '2019-08-01 00:15:00'
    bounds = locate.bounds(['2019-08-01 00:00:00'],
            dt.timedelta(minutes=15))
    result = locate.in_bounds(bounds, time)
    expect = [False]
    np.testing.assert_array_equal(expect, result)
