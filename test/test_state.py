import pytest
import datetime as dt
import numpy as np
from forest import db


@pytest.mark.parametrize("left,right,expect", [
        (db.State(), db.State(), True),

        (db.State(valid_time="2019-01-01 00:00:00"),
            db.State(valid_time=dt.datetime(2019, 1, 1)), True),

        (db.State(initial_time="2019-01-01 00:00:00"),
            db.State(initial_time=dt.datetime(2019, 1, 1)), True),

        (db.State(initial_times=np.array([
                "2019-01-01 00:00:00"], dtype='datetime64[s]')),
                db.State(initial_times=["2019-01-01 00:00:00"]), True),

        (db.State(initial_times=[]),
            db.State(initial_times=["2019-01-01 00:00:00"]), False),

        (db.State(valid_times=np.array([
                "2019-01-01 00:00:00"], dtype='datetime64[s]')),
                db.State(valid_times=["2019-01-01 00:00:00"]), True),

        (db.State(pressure=1000.001), db.State(pressure=1000.0001), True),
        (db.State(pressure=1000.001), db.State(pressure=900), False),
        (db.State(pressures=np.array([1000.001, 900])),
            db.State(pressures=[1000.0001, 900]), True),
        (db.State(pressures=[1000.001, 900]),
            db.State(pressures=[900, 900]), False),
        (db.State(variables=[]), db.State(), False),
        (db.State(variables=["a", "b"]), db.State(), False),
        (db.State(), db.State(variables=["a", "b"]), False),
        (db.State(variables=["a", "b"]), db.State(variables=["a", "b"]), True),
        (db.State(variables=np.array(["a", "b"])),
            db.State(variables=["a", "b"]), True),
        ])
def test_equality_and_not_equality(left, right, expect):
    assert (left == right) == expect
    assert (left != right) == (not expect)
