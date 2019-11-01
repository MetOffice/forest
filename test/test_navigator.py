from unittest.mock import Mock, call, patch, sentinel
import datetime as dt
import numpy as np
import pytest

import forest
import forest.navigate as navigate


@patch('forest.navigate.Navigator._from_group')
def test_Navigator_init(from_group):
    from_group.side_effect = [sentinel.nav1, sentinel.nav2]
    group1 = Mock(pattern='pattern1')
    group2 = Mock(pattern='pattern2')
    config = Mock(file_groups=[group1, group2])

    navigator = navigate.Navigator(config, sentinel.database)

    assert from_group.mock_calls == [call(group1, sentinel.database),
                                     call(group2, sentinel.database)]
    assert navigator._navigators == {'pattern1': sentinel.nav1,
                                     'pattern2': sentinel.nav2}


def test_Navigator_from_group__use_database():
    navigator = navigate.Navigator._from_group(Mock(locator='database'),
                                               sentinel.database)
    assert navigator == sentinel.database


@patch('forest.navigate.FileSystemNavigator.from_file_type')
@patch('forest.navigate.Navigator._expand_paths')
def test_Navigator_from_group__use_paths(expand_paths, from_file_type):
    expand_paths.return_value = sentinel.paths
    from_file_type.return_value = sentinel.navigator
    group = Mock(locator='not-a-database', directory=sentinel.directory,
                 pattern=sentinel.pattern, file_type=sentinel.file_type)

    navigator = navigate.Navigator._from_group(group, sentinel.database)

    expand_paths.assert_called_once_with(sentinel.directory, sentinel.pattern)
    from_file_type.assert_called_once_with(sentinel.paths, sentinel.file_type)
    assert navigator == sentinel.navigator


@patch('glob.glob')
@patch('os.path.expanduser')
def test_Navigator_expand_paths__no_dir(expanduser, glob):
    expanduser.return_value = sentinel.expanded
    glob.return_value = sentinel.paths

    paths = navigate.Navigator._expand_paths(None, 'my-pattern')

    expanduser.assert_called_once_with('my-pattern')
    glob.assert_called_once_with(sentinel.expanded)
    assert paths == sentinel.paths


@patch('glob.glob')
@patch('os.path.expanduser')
@patch('os.path.join')
def test_Navigator_expand_paths__with_dir(join, expanduser, glob):
    join.return_value = sentinel.joined
    expanduser.return_value = sentinel.expanded
    glob.return_value = sentinel.paths

    paths = navigate.Navigator._expand_paths('my-dir', 'my-pattern')

    join.assert_called_once_with('my-dir', 'my-pattern')
    expanduser.assert_called_once_with(sentinel.joined)
    glob.assert_called_once_with(sentinel.expanded)
    assert paths == sentinel.paths


@pytest.mark.skip("use real unified model file")
def test_unified_model_navigator():
    paths = ["unified.nc"]
    navigator = forest.navigate.FileSystem(
            paths,
            coordinates=forest.unified_model.Coordinates())
    result = navigator.initial_times("*.nc")
    expect = []
    assert expect == result


@pytest.fixture
def rdt_navigator():
    paths = ["/some/file_201901010000.json"]
    return forest.navigate.FileSystemNavigator.from_file_type(paths, "rdt")


def test_rdt_navigator_valid_times_given_single_file(rdt_navigator):
    paths = ["/some/rdt_201901010000.json"]
    navigator = forest.navigate.FileSystemNavigator.from_file_type(paths, "rdt")
    actual = navigator.valid_times("*.json", None, None)
    expected = ["2019-01-01 00:00:00"]
    assert actual == expected


def test_rdt_navigator_valid_times_given_multiple_files(rdt_navigator):
    paths = [
            "/some/rdt_201901011200.json",
            "/some/rdt_201901011215.json",
            "/some/rdt_201901011230.json"
    ]
    navigator = forest.navigate.FileSystemNavigator.from_file_type(paths, "rdt")
    actual = navigator.valid_times(paths[1], None, None)
    expected = ["2019-01-01 12:15:00"]
    np.testing.assert_array_equal(actual, expected)


def test_rdt_navigator_variables(rdt_navigator):
    assert rdt_navigator.variables("*.json") == ["RDT"]


def test_rdt_navigator_initial_times(rdt_navigator):
    assert rdt_navigator.initial_times("*.json") == ["2019-01-01 00:00:00"]


def test_rdt_navigator_pressures(rdt_navigator):
    assert rdt_navigator.pressures("*.json", None, None) == []
