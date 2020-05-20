from unittest.mock import Mock, sentinel
import pytest
import datetime as dt
import cftime
import numpy as np
import re

import forest.db.database as database
import forest.mark


def _create_db():
    cursor = Mock()
    cursor.fetchall.return_value = [(sentinel.value1,), (sentinel.value2,)]
    connection = Mock()
    connection.cursor.return_value = cursor
    db = database.Database(connection)
    cursor.reset_mock()
    return db


def _assert_query_and_params(db, expected_query, expected_params):
    db.cursor.execute.assert_called_once()
    args, kwargs = db.cursor.execute.call_args
    query, params = args
    assert_query_equal(query, expected_query)
    assert params == expected_params
    assert kwargs == {}


def assert_query_equal(left, right):
    left, right = single_spaced(left), single_spaced(right)
    assert left == right


def single_spaced(query):
    query = query.replace("\n", "")
    return re.sub(r'\s+', ' ', query).strip()


def test_Database_valid_times__defaults():
    db = _create_db()

    valid_times = db.valid_times(None, None, None)

    _assert_query_and_params(db, 'SELECT time.value FROM time',
                             {'pattern': None, 'variable': None,
                              'initial_time': None})
    assert valid_times == [sentinel.value1, sentinel.value2]


def test_Database_valid_times__all_args():
    db = _create_db()

    valid_times = db.valid_times(sentinel.pattern, sentinel.variable,
                                 dt.datetime(2020, 1, 1))

    _assert_query_and_params(
        db, 'SELECT time.value FROM time'
            ' JOIN variable_to_time AS vt ON vt.time_id = time.id'
            ' JOIN variable AS v ON vt.variable_id = v.id'
            ' JOIN file ON v.file_id = file.id'
            ' WHERE file.reference = :initial_time'
            ' AND file.name GLOB :pattern AND v.name = :variable',
        {'pattern': sentinel.pattern, 'variable': sentinel.variable,
         'initial_time': "2020-01-01 00:00:00"})
    assert valid_times == [sentinel.value1, sentinel.value2]


@pytest.mark.parametrize("pattern, variable, initial_time, expect", [
    (None, None, None, """
         SELECT time.value FROM time
    """),
    (sentinel.pattern, None, None, """
         SELECT time.value FROM time
           JOIN variable_to_time AS vt ON vt.time_id = time.id
           JOIN variable AS v ON vt.variable_id = v.id
           JOIN file ON v.file_id = file.id
          WHERE file.name GLOB :pattern
    """),
    (None, sentinel.variable, None, """
         SELECT time.value FROM time
           JOIN variable_to_time AS vt ON vt.time_id = time.id
           JOIN variable AS v ON vt.variable_id = v.id
           JOIN file ON v.file_id = file.id
          WHERE v.name = :variable
    """),
    (sentinel.pattern, sentinel.variable, None, """
         SELECT time.value FROM time
           JOIN variable_to_time AS vt ON vt.time_id = time.id
           JOIN variable AS v ON vt.variable_id = v.id
           JOIN file ON v.file_id = file.id
          WHERE file.name GLOB :pattern AND v.name = :variable
    """),
    (sentinel.pattern, sentinel.variable, sentinel.initial_time, """
         SELECT time.value FROM time
           JOIN variable_to_time AS vt ON vt.time_id = time.id
           JOIN variable AS v ON vt.variable_id = v.id
           JOIN file ON v.file_id = file.id
          WHERE file.reference = :initial_time
            AND file.name GLOB :pattern AND v.name = :variable
    """),
])
def test_valid_times_query(pattern, variable, initial_time, expect):
    result = database.Database.valid_times_query(pattern, variable, initial_time)
    assert_query_equal(expect, result)


@pytest.mark.parametrize("pattern, variable, initial_time, expect", [
    (None, None, None, """
         SELECT DISTINCT value FROM pressure
          ORDER BY value
    """),
    (sentinel.pattern, None, None, """
         SELECT DISTINCT pressure.value FROM pressure
           JOIN variable_to_pressure AS vp ON vp.pressure_id = pressure.id
           JOIN variable AS v ON v.id = vp.variable_id
           JOIN file ON v.file_id = file.id
          WHERE file.name GLOB :pattern
          ORDER BY value
    """),
    (None, sentinel.variable, None, """
         SELECT DISTINCT pressure.value FROM pressure
           JOIN variable_to_pressure AS vp ON vp.pressure_id = pressure.id
           JOIN variable AS v ON v.id = vp.variable_id
           JOIN file ON v.file_id = file.id
          WHERE v.name = :variable
          ORDER BY value
    """),
    (sentinel.pattern, sentinel.variable, None, """
         SELECT DISTINCT pressure.value FROM pressure
           JOIN variable_to_pressure AS vp ON vp.pressure_id = pressure.id
           JOIN variable AS v ON v.id = vp.variable_id
           JOIN file ON v.file_id = file.id
          WHERE v.name = :variable AND file.name GLOB :pattern
          ORDER BY value
    """),
    (sentinel.pattern, sentinel.variable, sentinel.initial_time, """
         SELECT DISTINCT pressure.value FROM pressure
           JOIN variable_to_pressure AS vp ON vp.pressure_id = pressure.id
           JOIN variable AS v ON v.id = vp.variable_id
           JOIN file ON v.file_id = file.id
          WHERE v.name = :variable
            AND file.name GLOB :pattern
            AND file.reference = :initial_time
          ORDER BY value
    """),
])
def test_pressures_query(pattern, variable, initial_time, expect):
    result = database.Database.pressures_query(pattern, variable, initial_time)
    assert_query_equal(expect, result)


def test_Database_pressures__defaults():
    db = _create_db()

    pressures = db.pressures()

    _assert_query_and_params(db, 'SELECT DISTINCT value FROM pressure'
                                 ' ORDER BY value',
                             {'pattern': None, 'variable': None,
                              'initial_time': None})
    assert pressures == [sentinel.value1, sentinel.value2]


def test_Database_pressures__all_args():
    db = _create_db()

    pressures = db.pressures(sentinel.pattern, sentinel.variable,
                             dt.datetime(2020, 1, 1))

    _assert_query_and_params(
        db, 'SELECT DISTINCT pressure.value FROM pressure'
            ' JOIN variable_to_pressure AS vp ON vp.pressure_id = pressure.id'
            ' JOIN variable AS v ON v.id = vp.variable_id'
            ' JOIN file ON v.file_id = file.id'
            ' WHERE v.name = :variable AND file.name GLOB :pattern'
            ' AND file.reference = :initial_time'
            ' ORDER BY value',
        {'pattern': sentinel.pattern, 'variable': sentinel.variable,
         'initial_time': "2020-01-01 00:00:00"})
    assert pressures == [sentinel.value1, sentinel.value2]



@pytest.mark.parametrize("initial_time", [
    pytest.param(dt.datetime(2020, 1, 1), id="datetime"),
    pytest.param(cftime.DatetimeGregorian(2020, 1, 1), id="cftime"),
    pytest.param(np.datetime64("2020-01-01", "ns"), id="np.datetime64"),
])
def test_Database_valid_times_given_datetime_like_objects(initial_time):
    initial_datetime = dt.datetime(2020, 1, 1)
    valid_times = [dt.datetime(2020, 1, 1, 12)]
    db = database.Database.connect(":memory:")
    db.insert_file_name("file.nc", initial_datetime)
    db.insert_times("file.nc", "air_temperature", valid_times)
    result = db.valid_times("file.nc", "air_temperature", initial_time)
    expect = ["2020-01-01 12:00:00"]
    assert expect == result


@pytest.mark.parametrize("time", [
    pytest.param(dt.datetime(2020, 1, 1), id="datetime"),
    pytest.param(cftime.DatetimeGregorian(2020, 1, 1), id="cftime"),
    pytest.param(np.datetime64("2020-01-01", "ns"), id="np.datetime64"),
])
def test_sanitize_datetime_like_objects(time):
    assert forest.mark.sanitize_time(time) == "2020-01-01 00:00:00"
