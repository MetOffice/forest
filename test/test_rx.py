from unittest.mock import Mock, sentinel, call
import forest.rx


def test_distinct_given_true_comparator():
    comparator = Mock(return_value=True)
    listener = Mock()
    stream = forest.rx.Stream()
    stream.distinct(comparator).map(listener)
    stream.notify(sentinel.first)
    stream.notify(sentinel.second)
    listener.assert_called_once_with(sentinel.first)


def test_distinct_given_false_comparator():
    comparator = Mock(return_value=False)
    listener = Mock()
    stream = forest.rx.Stream()
    stream.distinct(comparator).map(listener)
    stream.notify(sentinel.first)
    stream.notify(sentinel.second)
    assert listener.call_count == 2, "listener should only be called twice"
    listener.assert_has_calls([
        call(sentinel.first),
        call(sentinel.second)
    ])
