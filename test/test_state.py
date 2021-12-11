import pytest
import datetime as dt
import numpy as np
import pandas as pd
import cftime
import bokeh.palettes
import forest.state
from forest.db.control import time_array_equal
import forest.db.control


@pytest.mark.parametrize(
    "left,right,expect",
    [
        (forest.db.control.State(), forest.db.control.State(), True),
        (
            forest.db.control.State(valid_time="2019-01-01 00:00:00"),
            forest.db.control.State(valid_time=dt.datetime(2019, 1, 1)),
            True,
        ),
        (
            forest.db.control.State(initial_time="2019-01-01 00:00:00"),
            forest.db.control.State(initial_time=dt.datetime(2019, 1, 1)),
            True,
        ),
        (
            forest.db.control.State(
                initial_times=np.array(
                    ["2019-01-01 00:00:00"], dtype="datetime64[s]"
                )
            ),
            forest.db.control.State(initial_times=["2019-01-01 00:00:00"]),
            True,
        ),
        (
            forest.db.control.State(initial_times=[]),
            forest.db.control.State(initial_times=["2019-01-01 00:00:00"]),
            False,
        ),
        (
            forest.db.control.State(
                valid_times=np.array(
                    ["2019-01-01 00:00:00"], dtype="datetime64[s]"
                )
            ),
            forest.db.control.State(valid_times=["2019-01-01 00:00:00"]),
            True,
        ),
        (
            forest.db.control.State(pressure=1000.001),
            forest.db.control.State(pressure=1000.0001),
            True,
        ),
        (
            forest.db.control.State(pressure=1000.001),
            forest.db.control.State(pressure=900),
            False,
        ),
        (
            forest.db.control.State(pressures=np.array([1000.001, 900])),
            forest.db.control.State(pressures=[1000.0001, 900]),
            True,
        ),
        (
            forest.db.control.State(pressures=[1000.001, 900]),
            forest.db.control.State(pressures=[900, 900]),
            False,
        ),
        (
            forest.db.control.State(variables=[]),
            forest.db.control.State(),
            False,
        ),
        (
            forest.db.control.State(variables=["a", "b"]),
            forest.db.control.State(),
            False,
        ),
        (
            forest.db.control.State(),
            forest.db.control.State(variables=["a", "b"]),
            False,
        ),
        (
            forest.db.control.State(variables=["a", "b"]),
            forest.db.control.State(variables=["a", "b"]),
            True,
        ),
        (
            forest.db.control.State(variables=np.array(["a", "b"])),
            forest.db.control.State(variables=["a", "b"]),
            True,
        ),
    ],
)
def test_equality_and_not_equality(left, right, expect):
    assert (left == right) == expect
    assert (left != right) == (not expect)


def test_state_equality_valueerror_lengths_must_match():
    """should return False if lengths do not match"""
    valid_times = (
        pd.date_range("2020-01-01", periods=2),
        pd.date_range("2020-01-01", periods=3),
    )
    left = forest.db.control.State(valid_times=valid_times[0])
    right = forest.db.control.State(valid_times=valid_times[1])
    assert (left == right) == False


def test_time_array_equal():
    left = pd.date_range("2020-01-01", periods=2)
    right = pd.date_range("2020-01-01", periods=3)
    assert time_array_equal(left, right) == False


def test_valueerror_lengths_must_match():
    a = ["2020-01-01T00:00:00Z"]
    b = [
        "2020-02-01T00:00:00Z",
        "2020-02-02T00:00:00Z",
        "2020-02-03T00:00:00Z",
    ]
    with pytest.raises(ValueError):
        pd.to_datetime(a) == pd.to_datetime(b)


@pytest.mark.parametrize(
    "left,right,expect",
    [
        pytest.param(
            [
                cftime.DatetimeGregorian(2020, 1, 1),
                cftime.DatetimeGregorian(2020, 1, 2),
                cftime.DatetimeGregorian(2020, 1, 3),
            ],
            pd.date_range("2020-01-01", periods=3),
            True,
            id="gregorian/pandas same values",
        ),
        pytest.param(
            [
                cftime.DatetimeGregorian(2020, 2, 1),
                cftime.DatetimeGregorian(2020, 2, 2),
                cftime.DatetimeGregorian(2020, 2, 3),
            ],
            pd.date_range("2020-01-01", periods=3),
            False,
            id="gregorian/pandas same length different values",
        ),
    ],
)
def test_time_array_equal_mixed_types(left, right, expect):
    assert time_array_equal(left, right) == expect


def test_dataclass_state_default():
    state = forest.state.State()
    names = list(sorted(bokeh.palettes.all_palettes.keys()))
    numbers = list(sorted(bokeh.palettes.all_palettes["Viridis"].keys()))
    assert state.bokeh.html_loaded == False  # noqa: E712
    assert state.borders.line_color == "black"
    assert state.borders.visible == False  # noqa: E712
    assert state.colorbar.name == "Viridis"
    assert state.colorbar.names == names
    assert state.colorbar.number == 256
    assert state.colorbar.numbers == numbers
    assert state.colorbar.reverse == False  # noqa: E712
    assert state.colorbar.invisible_min == False  # noqa: E712
    assert state.colorbar.invisible_max == False  # noqa: E712
    assert state.colorbar.limits.origin == "column_data_source"
    assert state.colorbar.limits.user.low == 0
    assert state.colorbar.limits.user.high == 1
    assert state.colorbar.limits.column_data_source.low == 0
    assert state.colorbar.limits.column_data_source.high == 1
    assert state.pattern is None
    assert state.variable is None
    assert state.initial_time == dt.datetime(1970, 1, 1)
    assert state.valid_time == dt.datetime(1970, 1, 1)
    assert state.pressure == 0
    assert state.variables == []
    assert state.initial_times == []
    assert state.valid_times == []
    assert state.pressures == []
    assert state.tile.name == "Open street map"
    assert state.tile.labels == False  # noqa: E712
    assert state.tools.profile == False  # noqa: E712
    assert state.tools.time_series == False  # noqa: E712
    assert state.layers.figures == 1
    assert state.layers.index == {}
    assert state.layers.active == []
    assert state.layers.mode.state == "add"
    assert state.layers.mode.index == 0
    assert state.position.x == 0
    assert state.position.y == -1e9
    assert state.presets.active == 0
    assert state.presets.labels == {}
    assert state.presets.meta == {}
