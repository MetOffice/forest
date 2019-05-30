import unittest
import numpy as np
import forest.util


class TestRadarColors(unittest.TestCase):
    def test_radar_colors(self):
        result = forest.util.radar_colors()
        expect = np.array([
            (255, 255, 255, 0),
            (122, 147, 212, 0.9 * 255),
            (82, 147, 212, 0.95 * 255),
            (39, 106, 188, 255),
            (31, 201, 26, 255),
            (253, 237, 57, 255),
            (245, 152, 0, 255),
            (235, 47, 26, 255),
            (254, 92, 252, 255),
            (220, 220, 220, 255),
        ]) / 255.
        self.assert_array_almost_equal(expect, result)

    def assert_array_almost_equal(self, expect, result):
        try:
            np.testing.assert_array_almost_equal(expect, result)
        except AssertionError as e:
            expect, result = np.asarray(expect), np.asarray(result)
            pts = np.where(np.abs(expect - result) > 0.01)
            if len(expect[pts]) > 0:
                msg = e.args[0]
                msg += "\n pts = {}".format(pts)
                msg += "\n x[pts] != y[pts]: {} != {}".format(expect[pts], result[pts])
                e.args = (msg,)
            raise e

