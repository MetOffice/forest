from unittest.mock import Mock
import forest.reusable_pool


def test_acquire_calls_prepare():
    obj = Mock()
    factory = Mock(return_value=obj)
    pool = forest.reusable_pool.ReusablePool(factory)
    pool.acquire()
    obj.prepare.assert_called_once_with()


def test_release_calls_reset():
    obj = Mock()
    pool = forest.reusable_pool.ReusablePool(None)
    pool.release(obj)
    obj.reset.assert_called_once_with()
