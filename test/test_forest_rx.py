import unittest
from forest import rx


class TestCombineLatest(unittest.TestCase):
    def test_combine_latest(self):
        stream_1 = rx.Stream()
        stream_2 = rx.Stream()
        observer = unittest.mock.Mock()
        combined = rx.combine_latest(stream_1, stream_2)
        combined.subscribe(observer)
        stream_1.emit(1)
        observer.assert_called_once_with((1, None))

    def test_combine_latest_given_multiple_emits(self):
        stream_1 = rx.Stream()
        stream_2 = rx.Stream()
        observer = unittest.mock.Mock()
        combined = rx.combine_latest(stream_1, stream_2)
        combined.subscribe(observer)
        stream_1.emit(1)
        stream_2.emit(4)
        stream_1.emit(0)
        expect = [(1, None), (1, 4), (0, 4)]
        calls = [unittest.mock.call(args)
                 for args in expect]
        observer.assert_has_calls(calls)

    def test_subscribe(self):
        observer = unittest.mock.Mock()
        stream = rx.Stream()
        stream.subscribe(observer)
        stream.emit((1, None))
        observer.assert_called_once_with((1, None))

