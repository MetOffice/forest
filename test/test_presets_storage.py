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


def test_storage_middleware_given_on_save(store):
    storage = presets.Storage()
    action = presets.on_save("label")
    middleware = presets.Middleware(storage)
    list(middleware(store, action))
    assert storage.load("label") == {}


def test_storage_middleware_given_on_save_copies_colorbar():
    store = redux.Store(presets.reducer, initial_state={"colorbar": {"K": "V"}})
    storage = presets.Storage()
    action = presets.on_save("label")
    middleware = presets.Middleware(storage)
    list(middleware(store, action))
    store.state["colorbar"]["K"] = "T"
    assert storage.load("label") == {"K": "V"}
