import unittest
import unittest.mock
import datetime as dt
import main


class TestRunControls(unittest.TestCase):
    def setUp(self):
        self.fixture = main.RunControls([])

    def test_constructor(self):
        result = self.fixture.initial_times
        expect = []
        self.assertEqual(expect, result)

    def test_constructor_given_datetimes(self):
        datetime = dt.datetime(2019, 1, 1, 12)
        date = dt.date(2019, 1, 1)
        time = dt.time(12)
        fixture = main.RunControls([datetime])
        self.assertEqual(fixture.initial_times, [datetime])
        self.assertEqual(fixture.initial_dates, [date])
        self.assertEqual(fixture.times(date), [time])

    def test_announce_given_date_then_time(self):
        listener = unittest.mock.Mock()
        self.fixture.subscribe(listener)
        self.fixture.on_date(None, None, dt.date(2019, 1, 1))
        self.fixture.on_time("12:00")
        expect = dt.datetime(2019, 1, 1, 12)
        listener.assert_called_once_with(expect)

    def test_announce_given_time_then_date(self):
        listener = unittest.mock.Mock()
        self.fixture.subscribe(listener)
        self.fixture.on_time("00:00")
        self.fixture.on_date(None, None, dt.date(2018, 2, 3))
        expect = dt.datetime(2018, 2, 3, 0)
        listener.assert_called_once_with(expect)
