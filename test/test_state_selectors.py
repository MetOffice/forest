import pytest
from forest import selectors, db


@pytest.mark.parametrize("state,expect", [
    ({}, None),
    ({"pressure": 1000.}, 1000.),
    (db.State(pressure=750.), 750.)
])
def test_pressure(state, expect):
    assert selectors.pressure(state) == expect
