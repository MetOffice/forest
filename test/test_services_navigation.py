from unittest.mock import Mock, sentinel
import forest.services


def test_navigation_service():
    """experimental API"""
    dataset_name = "undefined"
    navigator = forest.services.navigation.get_navigator(dataset_name)
    assert isinstance(navigator, forest.services.NullNavigator)


def test_navigation_locator_provide():
    """Delegate to Dataset.navigator()"""
    dataset = Mock()
    dataset.navigator.return_value = sentinel.navigator
    forest.services.navigation.add_dataset("name", dataset)
    navigator = forest.services.navigation.get_navigator("name")
    assert navigator == sentinel.navigator


def test_null_navigator():
    """Navigator API"""
    navigator = forest.services.NullNavigator()
    pattern, variable, initial_time = None, None, None
    assert navigator.variables(pattern) == []
    assert navigator.initial_times(pattern, variable) == []
    assert navigator.valid_times(pattern, variable, initial_time) == []
    assert navigator.pressures(pattern, variable, initial_time) == []
