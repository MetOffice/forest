import unittest
import array
from collections import namedtuple
import numpy as np
from forest import geography
import forest


Extent = namedtuple("Extent", ["x_start", "x_end",
                               "y_start", "y_end"])


class TestBoundingSquare(unittest.TestCase):
    def test_bounding_square_preserves_unit_square(self):
        self.check_bounding_square((0, 0, 1, 1),
                                   (0, 0, 1, 1))

    def test_bounding_square_given_wide_rectangle(self):
        self.check_bounding_square((0, 0, 2, 1),
                                   (0, -0.5, 2, 1.5))

    def check_bounding_square(self, given, expect):
        result = forest.bounding_square(*given)
        self.assertEqual(expect, result)


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
