from unittest.mock import Mock, sentinel
from forest.object_pool import ObjectPool


def test_object_pool_acquire():
    factory = Mock(return_value=sentinel.object)
    pool = ObjectPool(factory)
    result = pool.acquire()
    assert result == sentinel.object
    factory.assert_called_once_with()


def test_object_pool_acquire_release():
    factory = Mock(return_value=sentinel.object)
    pool = ObjectPool(factory)
    obj = pool.acquire()
    pool.release(obj)
    result = pool.acquire()
    assert result == sentinel.object
    factory.assert_called_once_with()
