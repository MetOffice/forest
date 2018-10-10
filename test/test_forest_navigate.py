import unittest
import datetime as dt
import bokeh.models
import forest


def date_range(start, periods, interval):
    return [start + i * interval for i in range(periods)]


def bounded_add(minimum, maximum):
    def add(a, b):
        if (a + b) > maximum:
            return maximum
        if (a + b) < minimum:
            return minimum
        return a + b
    return add


class TestNextTime(unittest.TestCase):
    def test_next_time(self):
        stream = forest.Stream()
        scanned = stream.scan(0, bounded_add(0, 2))
        stream.emit(+1)
        stream.emit(+1)
        stream.emit(+1)
        stream.emit(+1)
        self.assertEqual(scanned.state, 2)

    def test_bounded_add_exceeding_maximum_returns_maximum(self):
        add = bounded_add(0, 3)
        result = add(2, 2)
        expect = 3
        self.assertEqual(expect, result)

    def test_bounded_add_exceeding_minimum_returns_minimum(self):
        add = bounded_add(0, 3)
        result = add(1, -2)
        expect = 0
        self.assertEqual(expect, result)


class TestStream(unittest.TestCase):
    def setUp(self):
        self.stream = forest.rx.Stream()
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


class TestTimeControl(unittest.TestCase):
    """Forecast/real time navigation tools"""
    def setUp(self):
        periods = 12
        interval = dt.timedelta(hours=3)
        first_start = dt.datetime(2018, 1, 1, 12, 0, 0)
        first_run = date_range(first_start, periods, interval)
        self.second_start = dt.datetime(2018, 1, 2, 0, 0, 0)
        second_run = date_range(self.second_start, periods, interval)
        forecast_times = [first_run, second_run]
        self.control = forest.TimeControl(forecast_times)

    def test_model_start_time(self):
        result = self.control.model_start_time
        expect = dt.datetime(2018, 1, 1, 12, 0, 0)
        self.assertEqual(expect, result)

    def test_next_forecast_increases_forecast_length(self):
        self.control.next_forecast()
        result = self.control.forecast_length
        expect = dt.timedelta(hours=3)
        self.assertEqual(expect, result)

    def test_previous_forecast_decreases_forecast_length(self):
        self.control.next_forecast()
        self.control.previous_forecast()
        result = self.control.forecast_length
        expect = dt.timedelta(hours=0)
        self.assertEqual(expect, result)

    def test_next_run(self):
        self.control.next_run()
        result = self.control.valid_time
        expect = self.second_start
        self.assertEqual(expect, result)

    def test_next_lead_time_keeps_valid_time_fixed(self):
        self.control.next_run()
        self.control.next_lead_time()
        result = self.control.valid_time
        expect = self.second_start
        self.assertEqual(expect, result)

    def test_next_lead_time_keeps_decrements_model_start_time(self):
        self.control.next_run()
        self.control.next_lead_time()
        result = self.control.model_start_time
        expect = dt.datetime(2018, 1, 1, 12, 0, 0)
        self.assertEqual(expect, result)

    def test_next_lead_time_raises_indexerror_if_on_first_run(self):
        with self.assertRaises(IndexError):
            self.control.next_lead_time()

    def test_next_run_raises_indexerror_if_on_last_run(self):
        self.control.most_recent_run()
        with self.assertRaises(IndexError):
            self.control.next_run()
