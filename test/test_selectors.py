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
