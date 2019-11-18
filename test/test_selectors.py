import pytest
import datetime as dt
from forest import selectors, db


@pytest.mark.parametrize("attr,state,expect", [
    ("pressure", {}, None),
    ("pressure", {"pressure": 1000.}, 1000.),
    ("pressure", db.State(pressure=750.), 750.),
    ("variable", {}, None),
    ("variable", {"variable": "air_temperature"}, "air_temperature"),
    ("variable", db.State(variable="mslp"), "mslp"),
    ("initial_time", {}, None),
    ("initial_time", {"initial_time": "2019-01-01 00:00:00"}, dt.datetime(2019, 1, 1)),
    ("initial_time", db.State(initial_time="2019-01-01 00:00:00"), dt.datetime(2019, 1, 1)),
    ("valid_time", {}, None),
    ("valid_time", {"valid_time": "2019-01-01 00:00:00"}, dt.datetime(2019, 1, 1)),
    ("valid_time", db.State(valid_time="2019-01-01 00:00:00"), dt.datetime(2019, 1, 1)),
    ("pressures", {}, None),
    ("pressures", {"pressure": 1000.}, None),
    ("pressures", {"pressures": [1000.]}, [1000.]),
    ("pressures", db.State(pressures=[750.]), [750.])
])
def test_selector_getattr(attr, state, expect):
    assert getattr(selectors.Selector(state), attr) == expect


@pytest.mark.parametrize("attr,state,expect", [
    ("pressures", {}, False),
    ("pressures", {"pressures": [1000, 900, 800]}, True),
    ("pressures", db.State(), False),
    ("pressures", db.State(pressures=[750.]), True),
    ])
def test_selector_defined(attr, state, expect):
    selector = selectors.Selector(state)
    assert selector.defined(attr) == expect
