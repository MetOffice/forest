import unittest
from forest import rx


class TestStream(unittest.TestCase):
    def setUp(self):
        self.stream = rx.Stream()
        self.subscriber = unittest.mock.Mock()

    def test_emit_notifies_subscribers(self):
        value = 0
        self.stream.register(self.subscriber)
        self.stream.emit(value)
        self.subscriber.notify.assert_called_once_with(value)

    def test_map(self):
        mapped = self.stream.map(lambda x: x + 1)
        mapped.register(self.subscriber)
        self.stream.emit(2)
        self.subscriber.notify.assert_called_once_with(3)

    def test_filter(self):
        filtered = self.stream.filter(lambda x: x > 5)
        filtered.register(self.subscriber)
        self.stream.emit(7)
        self.stream.emit(2)
        self.stream.emit(6)
        self.subscriber.notify.assert_called_once_with(2)

    def test_log_is_chainable(self):
        mapped = self.stream.log().map(lambda x: x * 2)
        mapped.register(self.subscriber)
        self.stream.emit(2)
        self.subscriber.notify.assert_called_once_with(4)

    def test_scan(self):
        scanned = self.stream.scan(0, lambda a, i: a + i)
        scanned.register(self.subscriber)
        self.stream.emit(7)
        self.stream.emit(2)
        self.stream.emit(6)
        expect = [unittest.mock.call(n) for n in [7, 9, 15]]
        self.subscriber.notify.assert_has_calls(expect)

    def test_unique(self):
        altered = self.stream.unique()
        altered.register(self.subscriber)
        self.stream.emit(1)
        self.stream.emit(2)
        self.stream.emit(2)
        self.stream.emit(2)
        self.stream.emit(3)
        self.stream.emit(3)
        expect = [unittest.mock.call(i) for i in [1, 2, 3]]
        self.subscriber.notify.assert_has_calls(expect)


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

    def test_combine_unsubscribe(self):
        stream_1 = rx.Stream()
        stream_2 = rx.Stream()
        observer = unittest.mock.Mock()
        combined = rx.combine_latest(stream_1, stream_2)
        unsubscribe = combined.subscribe(observer)
        stream_1.emit(1)
        stream_2.emit(4)
        stream_1.emit(0)
        unsubscribe()
        stream_2.emit(1)
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

    def test_unsubscribe(self):
        observer = unittest.mock.Mock()
        stream = rx.Stream()
        unsubscribe = stream.subscribe(observer)
        stream.emit((1, None))
        unsubscribe()
        stream.emit((2, None))
        observer.assert_called_once_with((1, None))
