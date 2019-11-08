from unittest.mock import Mock, sentinel
import re

import forest.db.database as database


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
    query = re.sub(r'\s+', ' ', query).strip()
    assert query == expected_query
    assert params == expected_params
    assert kwargs == {}


def test_Database_valid_times__defaults():
    db = _create_db()

    valid_times = db.valid_times()

    _assert_query_and_params(db, 'SELECT time.value FROM time',
                             {'pattern': None, 'variable': None,
                              'initial_time': None})
    assert valid_times == [sentinel.value1, sentinel.value2]


def test_Database_valid_times__all_args():
    db = _create_db()

    valid_times = db.valid_times(sentinel.pattern, sentinel.variable,
                                 sentinel.initial_time)

    _assert_query_and_params(
        db, 'SELECT time.value FROM time'
            ' JOIN variable_to_time AS vt ON vt.time_id = time.id'
            ' JOIN variable AS v ON vt.variable_id = v.id'
            ' JOIN file ON v.file_id = file.id'
            ' WHERE file.reference = :initial_time'
            ' AND file.name GLOB :pattern AND v.name = :variable',
        {'pattern': sentinel.pattern, 'variable': sentinel.variable,
         'initial_time':sentinel.initial_time})
    assert valid_times == [sentinel.value1, sentinel.value2]


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
                             sentinel.initial_time)

    _assert_query_and_params(
        db, 'SELECT DISTINCT pressure.value FROM pressure'
            ' JOIN variable_to_pressure AS vp ON vp.pressure_id = pressure.id'
            ' JOIN variable AS v ON v.id = vp.variable_id'
            ' JOIN file ON v.file_id = file.id'
            ' WHERE v.name = :variable AND file.name GLOB :pattern'
            ' AND file.reference = :initial_time'
            ' ORDER BY value',
        {'pattern': sentinel.pattern, 'variable': sentinel.variable,
         'initial_time':sentinel.initial_time})
    assert pressures == [sentinel.value1, sentinel.value2]
