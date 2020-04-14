import pytest
import bokeh.models
import forest.drivers
import forest.layers
import forest.view


@pytest.fixture
def factory():
    color_mapper = bokeh.models.LinearColorMapper()
    name = "unified_model"
    settings = {"pattern": "*.nc"}
    dataset = forest.drivers.get_dataset(name, settings)
    figures = []
    return forest.layers.Factory(dataset, color_mapper, figures)


@pytest.fixture
def pool(factory):
    return forest.layers.Pool(factory)


def test_pool(pool):
    layer = pool.acquire()
    pool.release(layer)
    assert isinstance(layer,
                      forest.view.AbstractMapView)


def test_pool_acquire_same_object(pool):
    layer_0 = pool.acquire()
    pool.release(layer_0)
    layer_1 = pool.acquire()
    assert id(layer_0) == id(layer_1)


def test_pool_acquire_different_objects(pool):
    layer_0 = pool.acquire()
    layer_1 = pool.acquire()
    assert id(layer_0) != id(layer_1)
