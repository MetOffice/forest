import pytest
import unittest.mock
import pygrib
from forest import nearcast


def make_open(names):
    # Simulate pygrib.open(path).select() -> messages
    def _open(path):
        messages = unittest.mock.Mock()
        messages.select.return_value = [
            {"name": name} for name in names
        ]
        return messages
    return _open


@pytest.mark.parametrize("names,expect", [
    (["A", "A", "A"], ["A"]),
    (["D", "C", "B", "A"], ["A", "B", "C", "D"]),
])
def test_navigator_variables(monkeypatch, names, expect):
    pattern = ""
    navigator = nearcast.Navigator(pattern)
    navigator.locator = unittest.mock.Mock()
    navigator.locator.find.return_value = ["some.grib"]
    monkeypatch.setattr(pygrib, "open", make_open(names))
    assert navigator.variables(pattern) == expect
