import pytest
import datetime as dt
import numpy as np
import pandas as pd
import cftime
from forest import db
from forest.db.control import time_array_equal


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
            db.State(variables=["a", "b"]), True)
        ])
def test_equality_and_not_equality(left, right, expect):
    assert (left == right) == expect
    assert (left != right) == (not expect)


def test_state_equality_valueerror_lengths_must_match():
    """should return False if lengths do not match"""
    valid_times = (
        pd.date_range("2020-01-01", periods=2),
        pd.date_range("2020-01-01", periods=3),
    )
    left = db.State(valid_times=valid_times[0])
    right = db.State(valid_times=valid_times[1])
    assert (left == right) == False


def test_time_array_equal():
    left = pd.date_range("2020-01-01", periods=2)
    right = pd.date_range("2020-01-01", periods=3)
    assert time_array_equal(left, right) == False


def test_valueerror_lengths_must_match():
    a = ["2020-01-01T00:00:00Z"]
    b = ["2020-02-01T00:00:00Z", "2020-02-02T00:00:00Z", "2020-02-03T00:00:00Z"]
    with pytest.raises(ValueError):
        pd.to_datetime(a) == pd.to_datetime(b)


def test_time_array_equal_mixed_types():
    left = [cftime.DatetimeGregorian(2020, 1, 1),
            cftime.DatetimeGregorian(2020, 1, 2),
            cftime.DatetimeGregorian(2020, 1, 3)]
    right = pd.date_range("2020-01-01", periods=3)
    assert time_array_equal(left, right) == True
