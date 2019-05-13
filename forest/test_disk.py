import unittest
import datetime as dt
import numpy as np
import netCDF4
import disk


class TestPattern(unittest.TestCase):
    def test_pattern_given_initial_time_and_length(self):
        initial = np.datetime64('2019-04-29 18:00', 's')
        length = np.timedelta64(33, 'h')
        pattern = "global_africa_{:%Y%m%dT%H%MZ}_umglaa_pa{:03d}.nc"
        result = disk.file_name(pattern, initial, length)
        expect = "global_africa_20190429T1800Z_umglaa_pa033.nc"
        self.assertEqual(expect, result)


@unittest.skip("green light")
class TestIndex(unittest.TestCase):
    def setUp(self):
        self.initial = dt.datetime(2019, 4, 30, 6)
        self.path = "/Users/andrewryan/cache/global_africa_20190430T0600Z_umglaa_pa009.nc"
        with netCDF4.Dataset(self.path) as dataset:
            var = dataset.variables["time"]
            self.times = netCDF4.num2date(var[:], units=var.units)
            self.pressures = dataset.variables["pressure"][:]

    def test_locate(self):
        time = dt.datetime(2019, 4, 30, 14, 1)
        pressure = 850.
        pts = disk.points(self.times, self.pressures, time, pressure)
        np.testing.assert_array_almost_equal(
                self.pressures[pts], pressure)
        result = self.times[pts][0]
        expect = time.replace(minute=0)
        self.assertEqual(expect, result)

    def test_lengths(self):
        result = disk.lengths(self.times, self.initial)
        self.assertEqual(len(result), 75)
        n = 25
        self.assertEqual(result.tolist(),
                ((n - 1) * [7]) +
                ((n - 1) * [8]) +
                ((n + 2) * [9]))
