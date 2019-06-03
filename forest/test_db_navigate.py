import unittest
import datetime as dt
from db import Observable


class InvalidTime(Exception):
    pass


class Control(Observable):
    def __init__(self, values, value=None):
        self.values = values
        self.value = value
        super().__init__()

    @property
    def current(self):
        return self.value

    def forward(self):
        if len(self.values) == 0:
            return
        if self.value is None:
            value = self.values[0]
        else:
            value = self.next_item(self.values, self.value)
        self.notify(value)
        self.value = value

    def backward(self):
        if len(self.values) == 0:
            return
        if self.value is None:
            value = self.values[-1]
        else:
            try:
                value = self.previous_item(self.values, self.value)
            except InvalidTime:
                return
        self.notify(value)
        self.value = value

    def reset(self, values):
        """Change available values to iterate over"""

    @staticmethod
    def next_item(values, value):
        index = values.index(value) + 1
        try:
            return values[index]
        except IndexError:
            raise InvalidTime('outside: {}'.format(values[-1]))

    @staticmethod
    def previous_item(values, value):
        index = values.index(value) - 1
        if index < 0:
            raise InvalidTime('outside: {}'.format(values[0]))
        return values[index]


class TestTimeNavigation(unittest.TestCase):
    def setUp(self):
        self.times = [
            dt.datetime(2019, 1, 1, 12),
            dt.datetime(2019, 1, 2, 0),
            dt.datetime(2019, 1, 2, 12)
        ]
        self.control = Control(self.times, self.times[0])

    def test_reset(self):
        control = Control([1, 2, 3])
        control.reset([3, 4, 5])

    def test_next_item(self):
        result = self.control.next_item(self.times, self.times[1])
        expect = self.times[2]
        self.assertEqual(expect, result)

    def test_next_raises_exception_if_outside_range(self):
        with self.assertRaises(InvalidTime):
            self.control.next_item(self.times, self.times[-1])

    def test_previous_item(self):
        result = self.control.previous_item(self.times, self.times[1])
        expect = self.times[0]
        self.assertEqual(expect, result)

    def test_previous_raises_exception_if_outside_range(self):
        with self.assertRaises(InvalidTime):
            self.control.previous_item(self.times, self.times[0])

    def test_observable(self):
        listener = unittest.mock.Mock()
        self.control.subscribe(listener)
        self.control.forward()
        listener.assert_called_once_with(self.times[1])

    def test_backward_calls_listener(self):
        listener = unittest.mock.Mock()
        self.control.value = self.times[2]
        self.control.subscribe(listener)
        self.control.backward()
        listener.assert_called_once_with(self.times[1])

    def test_backward_given_time_at_start_of_series_does_nothing(self):
        listener = unittest.mock.Mock()
        self.control.subscribe(listener)
        self.control.backward()
        self.assertFalse(listener.called)

    def test_forward_if_current_not_set_selects_first_item(self):
        control = Control([1, 2, 3])
        control.forward()
        result = control.current
        expect = 1
        self.assertEqual(expect, result)

    def test_forward_given_empty_list_does_nothing(self):
        control = Control([])
        control.forward()

    def test_backward_given_none_sets_last_item(self):
        control = Control([1, 2, 3])
        control.backward()
        result = control.current
        expect = 3
        self.assertEqual(expect, result)

    def test_backward_given_empty_list_does_nothing(self):
        control = Control([])
        control.backward()
