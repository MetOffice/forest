from unittest.mock import Mock, call, patch, sentinel

import numpy as np
import pytest

import forest.exceptions

import forest.navigate as navigate


def test_get_database_with_real_dependencies(tmpdir):
    # Note: needed to check os library imported
    database_path = str(tmpdir / "example.db")
    with pytest.raises(ValueError):
        forest.db.get_database(database_path)


@patch('forest.navigate.Navigator._from_group')
def test_Navigator_init(from_group):
    from_group.side_effect = [sentinel.nav1, sentinel.nav2]
    group1 = Mock(pattern='pattern1')
    group2 = Mock(pattern='pattern2')
    config = Mock(file_groups=[group1, group2])

    navigator = navigate.Navigator(config)

    color_mapper = None
    assert from_group.mock_calls == [call(group1, color_mapper),
                                     call(group2, color_mapper)]
    assert navigator._navigators == {'pattern1': sentinel.nav1,
                                     'pattern2': sentinel.nav2}


@patch('forest.navigate.FileSystemNavigator.from_file_type')
@patch('forest.navigate.Navigator._expand_paths')
def test_Navigator_from_group__use_paths(expand_paths, from_file_type):
    expand_paths.return_value = sentinel.paths
    from_file_type.return_value = sentinel.navigator
    group = Mock(locator='not-a-database',
                 pattern=sentinel.pattern, file_type=sentinel.file_type)

    navigator = navigate.Navigator._from_group(group)

    expand_paths.assert_called_once_with(sentinel.pattern)
    from_file_type.assert_called_once_with(sentinel.paths,
                                           sentinel.file_type,
                                           sentinel.pattern)
    assert navigator == sentinel.navigator


@patch('glob.glob')
@patch('os.path.expanduser')
def test_Navigator_expand_paths(expanduser, glob):
    expanduser.return_value = sentinel.expanded
    glob.return_value = sentinel.paths

    paths = navigate.Navigator._expand_paths('my-pattern')

    expanduser.assert_called_once_with('my-pattern')
    glob.assert_called_once_with(sentinel.expanded)
    assert paths == sentinel.paths


def test_Navigator_variables():
    sub_navigator = Mock()
    sub_navigator.variables.return_value = sentinel.variables
    navigator = Mock(_navigators={sentinel.pattern: sub_navigator})

    result = navigate.Navigator.variables(navigator, sentinel.pattern)

    sub_navigator.variables.assert_called_once_with(sentinel.pattern)
    assert result == sentinel.variables


def test_Navigator_initial_times():
    sub_navigator = Mock()
    sub_navigator.initial_times.return_value = sentinel.initial_times
    navigator = Mock(_navigators={sentinel.pattern: sub_navigator})

    result = navigate.Navigator.initial_times(navigator, sentinel.pattern,
                                              sentinel.variable)

    sub_navigator.initial_times.assert_called_once_with(
        sentinel.pattern, variable=sentinel.variable)
    assert result == sentinel.initial_times


def test_Navigator_valid_times():
    sub_navigator = Mock()
    sub_navigator.valid_times.return_value = sentinel.valid_times
    navigator = Mock(_navigators={sentinel.pattern: sub_navigator})

    result = navigate.Navigator.valid_times(navigator, sentinel.pattern,
                                            sentinel.variable,
                                            sentinel.initial_time)

    sub_navigator.valid_times.assert_called_once_with(
        sentinel.pattern, sentinel.variable, sentinel.initial_time)
    assert result == sentinel.valid_times


def test_Navigator_pressures():
    sub_navigator = Mock()
    sub_navigator.pressures.return_value = sentinel.pressures
    navigator = Mock(_navigators={sentinel.pattern: sub_navigator})

    result = navigate.Navigator.pressures(navigator, sentinel.pattern,
                                          sentinel.variable,
                                          sentinel.initial_time)

    sub_navigator.pressures.assert_called_once_with(
        sentinel.pattern, sentinel.variable, sentinel.initial_time)
    assert result == sentinel.pressures


def test_FileSystemNavigator_init():
    navigator = navigate.FileSystemNavigator(sentinel.paths, sentinel.coords)
    assert navigator.paths == sentinel.paths
    assert navigator.coordinates == sentinel.coords


@patch('forest.rdt.Coordinates')
def test_FileSystemNavigator_from_file_type__rdt(coordinates_cls):
    coordinates_cls.return_value = sentinel.coordinates

    navigator = navigate.FileSystemNavigator.from_file_type(sentinel.paths,
                                                            'rDt')

    coordinates_cls.assert_called_once_with()
    assert navigator.paths == sentinel.paths
    assert navigator.coordinates == sentinel.coordinates


@patch('forest.gridded_forecast.Navigator')
def test_FileSystemNavigator_from_file_type__griddedforecast(navigator_cls):
    navigator_cls.return_value = sentinel.navigator

    navigator = navigate.FileSystemNavigator.from_file_type(sentinel.paths,
                                                            'grIDdeDforeCAST')

    navigator_cls.assert_called_once_with(sentinel.paths)
    assert navigator == sentinel.navigator


def test_FileSystemNavigator_from_file_type__unrecognised():
    with pytest.raises(Exception, match='Unrecognised file type'):
        navigate.FileSystemNavigator.from_file_type(sentinel.paths, 'FOO')


def test_FileSystemNavigator_variables():
    navigator = Mock(paths=['first', 'second', 'third', 'air'])
    navigator.coordinates.variables.side_effect = [['one', 'two'], ['three'],
                                                   ['two', 'five']]

    variables = navigate.FileSystemNavigator.variables(navigator, '*ir*')

    assert navigator.coordinates.variables.mock_calls == [call('first'),
                                                          call('third'),
                                                          call('air')]
    assert variables == ['five', 'one', 'three', 'two']


def test_FileSystemNavigator_initial_times():
    navigator = Mock(paths=['first', 'second', 'third', 'air', 'bird'])
    navigator.coordinates.initial_time.side_effect = [
        'last', None, forest.exceptions.InitialTimeNotFound, 'first']

    variables = navigate.FileSystemNavigator.initial_times(navigator, '*ir*')

    assert navigator.coordinates.initial_time.mock_calls == [call('first'),
                                                             call('third'),
                                                             call('air'),
                                                             call('bird')]
    assert variables == ['first', 'last']


def test_FileSystemNavigator_valid_times():
    navigator = Mock(paths=['first', 'second', 'third', 'air', 'bird'])
    navigator.coordinates.valid_times.side_effect = [
        [2, 3, 4], None, forest.exceptions.ValidTimesNotFound, [1, 2, 5, 6]]

    valid_times = navigate.FileSystemNavigator.valid_times(
        navigator, '*ir*', sentinel.variable, sentinel.initial_time)

    assert navigator.coordinates.valid_times.mock_calls == [
        call('first', sentinel.variable),
        call('third', sentinel.variable),
        call('air', sentinel.variable),
        call('bird', sentinel.variable)]
    assert np.all(valid_times == [1, 2, 3, 4, 5, 6])


def test_FileSystemNavigator_valid_times__empty():
    navigator = Mock(paths=['first'])
    navigator.coordinates.valid_times.return_value = None

    valid_times = navigate.FileSystemNavigator.valid_times(
        navigator, '*ir*', sentinel.variable, sentinel.initial_time)

    assert navigator.coordinates.valid_times.mock_calls == [
        call('first', sentinel.variable)]
    assert valid_times == []


def test_FileSystemNavigator_pressures():
    navigator = Mock(paths=['first', 'second', 'third', 'air', 'bird'])
    navigator.coordinates.pressures.side_effect = [
        [2, 3, 4], None, forest.exceptions.PressuresNotFound, [1, 2, 5, 6]]

    pressures = navigate.FileSystemNavigator.pressures(navigator, '*ir*',
                                                       sentinel.variable,
                                                       sentinel.initial_time)

    assert navigator.coordinates.pressures.mock_calls == [
        call('first', sentinel.variable),
        call('third', sentinel.variable),
        call('air', sentinel.variable),
        call('bird', sentinel.variable)]
    assert np.all(pressures == [1, 2, 3, 4, 5, 6])


def test_FileSystemNavigator_pressures__empty():
    navigator = Mock(paths=['first'])
    navigator.coordinates.pressures.return_value = None

    pressures = navigate.FileSystemNavigator.pressures(navigator, '*ir*',
                                                       sentinel.variable,
                                                       sentinel.initial_time)

    assert navigator.coordinates.pressures.mock_calls == [
        call('first', sentinel.variable)]
    assert pressures == []
