import pytest
from forest import selectors, db


@pytest.mark.parametrize("attr,state,expect", [
    ("pressure", {}, None),
    ("pressure", {"pressure": 1000.}, 1000.),
    ("pressure", db.State(pressure=750.), 750.),
    ("pressures", {}, None),
    ("pressures", {"pressure": 1000.}, None),
    ("pressures", {"pressures": [1000.]}, [1000.]),
    ("pressures", db.State(pressures=[750.]), [750.])
])
def test_selector(attr, state, expect):
    assert getattr(selectors.Selector(state), attr) == expect
