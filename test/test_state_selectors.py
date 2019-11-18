from forest import selectors


def test_selector():
    assert selectors.pressure({}) is None
