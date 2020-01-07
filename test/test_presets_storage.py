import pytest
from forest import presets, colors, redux


@pytest.fixture
def store():
    return redux.Store(presets.reducer)


def test_storage_save_new():
    storage = presets.Storage()
    storage.save("label", {"key": "value"})
    assert storage.load("label") == {"key": "value"}


def test_storage_save_update():
    storage = presets.Storage()
    storage.save("label", {"key": "old"})
    storage.save("label", {"key": "new"})
    assert storage.load("label") == {"key": "new"}


def test_storage_save_copy():
    """Stored data must not be reference to mutable dict"""
    data = {"key": "old"}
    storage = presets.Storage()
    storage.save("label", data)
    data["key"] = "new"
    assert storage.load("label") == {"key": "old"}


def test_storage_load_copy():
    """Loaded data must not reference a mutable dict"""
    data = {"key": "old"}
    storage = presets.Storage()
    storage.save("label", data)
    loaded = storage.load("label")
    loaded["key"] = "mutate"  # Should not edit data inside Storage
    result = storage.load("label")
    assert result == {"key": "old"}


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


def test_storage_middleware_on_load(store):
    label = "label"
    settings = {"key": "value"}
    storage = presets.Storage()
    storage.save(label, settings)
    middleware = presets.Middleware(storage)
    action = presets.on_load(label)
    result = list(middleware(store, action))
    expect = [colors.set_colorbar(settings), action]
    assert expect == result
