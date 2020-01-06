import pytest
from forest import presets, redux


@pytest.fixture
def store():
    return redux.Store(presets.reducer)


def test_storage():
    storage = presets.Storage()
    storage.save("label", {"key": "value"})
    assert storage.load("label") == {"key": "value"}
    assert storage.labels() == ["label"]


def test_storage_middleware(store):
    storage = presets.Storage()
    storage.save("label", {"key": "value"})
    action = {"kind": "ANY"}
    middleware = presets.Middleware(storage)
    assert list(middleware(store, action)) == [action]
