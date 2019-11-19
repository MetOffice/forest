import pytest
import datetime as dt
import numpy as np
from forest import selectors, db


@pytest.mark.parametrize("attr,state,expect", [
    ("pressure", {}, None),
    ("pressure", {"pressure": 1000.}, 1000.),
    ("variable", {}, None),
    ("variable", {"variable": "air_temperature"}, "air_temperature"),
    ("initial_time", {}, None),
    ("initial_time", {"initial_time": "2019-01-01 00:00:00"}, dt.datetime(2019, 1, 1)),
    ("initial_time", {"initial_time": "2019-01-01T00:00:00"}, dt.datetime(2019, 1, 1)),
    ("initial_time", {"initial_time": np.datetime64("2019-01-01 00:00:00", "s")}, dt.datetime(2019, 1, 1)),
    ("valid_time", {}, None),
    ("valid_time", {"valid_time": "2019-01-01 00:00:00"}, dt.datetime(2019, 1, 1)),
    ("valid_time", {"valid_time": "2019-01-01T00:00:00"}, dt.datetime(2019, 1, 1)),
    ("valid_time", {"valid_time": np.datetime64("2019-01-01 00:00:00", "s")}, dt.datetime(2019, 1, 1)),
    ("pressures", {}, None),
    ("pressures", {"pressure": 1000.}, None),
    ("pressures", {"pressures": [1000.]}, [1000.]),
])
def test_selector_getattr(attr, state, expect):
    assert getattr(selectors.Selector(state), attr) == expect


@pytest.mark.parametrize("attr,state,expect", [
    ("pressures", {}, False),
    ("pressures", {"pressures": [1000, 900, 800]}, True),
    ])
def test_selector_defined(attr, state, expect):
    selector = selectors.Selector(state)
    assert selector.defined(attr) == expect


@pytest.mark.parametrize("time,expect", [
        (dt.datetime(2019, 1, 1), dt.datetime(2019, 1, 1)),
        ("2019-10-10 01:02:34", dt.datetime(2019, 10, 10, 1, 2, 34)),
        ("2019-10-10T01:02:34", dt.datetime(2019, 10, 10, 1, 2, 34)),
        (np.datetime64('2019-10-10T11:22:33'), dt.datetime(2019, 10, 10, 11, 22, 33)),
    ])
def test_to_datetime(time, expect):
    result = selectors.Selector.to_datetime(time)
    assert result == expect


def test_unsupported():
    with pytest.raises(Exception, match='Unknown value'):
        selectors.Selector.to_datetime(12)
