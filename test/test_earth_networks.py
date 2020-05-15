import pytest
import pandas as pd
import pandas.testing as pdt
import datetime as dt
import numpy as np
import glob
import forest.drivers
from forest.drivers import earth_networks


LINES = [
    "1,20190417T000001.440,+02.7514400,+031.9206400,-000001778,000,15635,007,001",
    "1,20190417T000001.093,+02.6388400,+031.9008800,+000002524,000,14931,007,012"
]


def test_earth_networks(tmpdir):
    path = str(tmpdir / "sample.txt")
    with open(path, "w") as stream:
        stream.writelines(LINES)

    loader = earth_networks.Loader()
    frame = loader.load([path])
    result = frame.iloc[0]
    atol = 0.000001
    if isinstance(result["date"], dt.datetime):
        # Pandas <0.25.x
        assert result["date"] == dt.datetime(2019, 4, 17, 0, 0, 1, 440000)
    else:
        # Pandas 1.0.x
        assert result["date"] == np.datetime64('2019-04-17T00:00:01.440000000')
    assert result["flash_type"] == "IC"
    assert abs(result["latitude"] - 2.75144) < atol
    assert abs(result["longitude"] - 31.92064) < atol


def test_dataset():
    dataset = forest.drivers.get_dataset("earth_networks")
    assert isinstance(dataset, forest.drivers.earth_networks.Dataset)


def get_navigator(settings):
    dataset = forest.drivers.get_dataset("earth_networks", settings)
    return dataset.navigator()


def test_dataset_navigator():
    navigator = get_navigator({"pattern": "*.txt"})
    assert isinstance(navigator, forest.drivers.earth_networks.Navigator)


def test_navigator_variables():
    navigator = earth_networks.Navigator([])
    assert set(navigator.variables(None)) == set([
        "Strike density (cloud-ground)",
        "Strike density (intra-cloud)",
        "Strike density (total)",
        "Time since flash (cloud-ground)",
        "Time since flash (intra-cloud)",
        "Time since flash (total)"
    ])
