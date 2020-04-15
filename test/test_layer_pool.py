import pytest
from unittest.mock import Mock, sentinel
import bokeh.models
import bokeh.plotting
import forest.drivers
import forest.layers
import forest.view


@pytest.fixture
def factory():
    color_mapper = bokeh.models.LinearColorMapper()
    name = "unified_model"
    settings = {"pattern": "*.nc"}
    dataset = forest.drivers.get_dataset(name, settings)
    figures = [bokeh.plotting.figure()]
    source_limits = Mock()
    opacity_slider = Mock()
    return forest.layers.Factory(dataset,
                                 color_mapper,
                                 figures,
                                 source_limits,
                                 opacity_slider)


@pytest.fixture
def pool(factory):
    return forest.layers.Pool(factory)


def test_pool(pool):
    layer = pool.acquire()
    pool.release(layer)
    assert isinstance(layer, forest.layers.Layer)


def test_pool_acquire_same_object(pool):
    layer_0 = pool.acquire()
    pool.release(layer_0)
    layer_1 = pool.acquire()
    assert id(layer_0) == id(layer_1)


def test_pool_acquire_different_objects(pool):
    layer_0 = pool.acquire()
    layer_1 = pool.acquire()
    assert id(layer_0) != id(layer_1)


def test_pool_map():
    """Apply function to objects in Pool"""
    method = Mock()
    factory = Mock()
    factory.return_value = sentinel.layer
    pool = forest.layers.Pool(factory)
    layer = pool.acquire()
    pool.release(layer)
    pool.map(method)
    method.assert_called_once_with(sentinel.layer)
