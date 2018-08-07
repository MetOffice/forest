import unittest
import array
from collections import namedtuple
import numpy as np
from forest import geography


class TestBoxSplit(unittest.TestCase):
    """Cut a piecewise linear coastline"""
    def test_box_split_given_vertical_line(self):
        x, y = [0.5, 0.5, 0.5], [-0.5, 0.5, 1.5]
        extent = (0, 1, 0, 1)
        result = list(geography.box_split(x, y, extent))
        expect = [
            [[0.5], [-0.5]],
            [[0.5], [0.5]],
            [[0.5], [1.5]]
        ]
        np.testing.assert_array_equal(result, expect)

    def test_box_split_given_vertical_line_outside_box(self):
        x, y = [2, 2, 2], [-0.5, 0.5, 1.5]
        extent = (0, 1, 0, 1)
        result = list(geography.box_split(x, y, extent))
        expect = [
            [x, y]
        ]
        np.testing.assert_array_equal(result, expect)


class TestCoastlines(unittest.TestCase):
    def test_global_110m_coastline(self):
        x, y = next(geography.coastlines())
        result = x[0], y[0]
        expect = -163.712896, -78.595667
        np.testing.assert_array_almost_equal(result, expect)

    def test_global_50m_coastline(self):
        x, y = next(geography.coastlines("50m"))
        result = x[0], y[0]
        expect = 180., -16.15293
        np.testing.assert_array_almost_equal(result, expect)

    def test_global_10m_coastline(self):
        x, y = next(geography.coastlines("10m"))
        result = x[0], y[0]
        expect = 59.916026, -67.400486
        np.testing.assert_array_almost_equal(result, expect)


Extent = namedtuple("Extent", ["x_start", "x_end",
                               "y_start", "y_end"])


class TestClipXY(unittest.TestCase):
    def setUp(self):
        self.x = np.array([1, 2, 3])
        self.y = np.array([4, 5, 6])

    def test_clip_xy(self):
        """helper to restrict coastlines to bokeh figure"""
        extent = Extent(-np.inf, np.inf, -np.inf, np.inf)
        expect = self.x, self.y
        self.check_clip_xy(self.x, self.y, extent, expect)

    def test_clip_xy_removes_points_left_of_extent(self):
        extent = Extent(1.5, np.inf, -np.inf, np.inf)
        expect = self.x[1:], self.y[1:]
        self.check_clip_xy(self.x, self.y, extent, expect)

    def test_clip_xy_removes_points_right_of_extent(self):
        extent = Extent(-np.inf, 2.5, -np.inf, np.inf)
        expect = self.x[:-1], self.y[:-1]
        self.check_clip_xy(self.x, self.y, extent, expect)

    def test_clip_xy_removes_points_below_extent(self):
        extent = Extent(-np.inf, np.inf, 4.5, np.inf)
        expect = self.x[1:], self.y[1:]
        self.check_clip_xy(self.x, self.y, extent, expect)

    def test_clip_xy_removes_points_above_extent(self):
        extent = Extent(-np.inf, np.inf, -np.inf, 5.5)
        expect = self.x[:-1], self.y[:-1]
        self.check_clip_xy(self.x, self.y, extent, expect)

    def test_clip_xy_supports_array_array(self):
        """should handle array.array returned by cartopy geometries"""
        x = array.array("f", [1, 2, 3])
        y = array.array("f", [4, 5, 6])
        extent = Extent(-np.inf, np.inf, -np.inf, 5.5)
        expect = x[:-1], y[:-1]
        self.check_clip_xy(x, y, extent, expect)

    def check_clip_xy(self, x, y, extent, expect):
        result = geography.clip_xy(x, y, extent)
        np.testing.assert_array_equal(result, expect)
