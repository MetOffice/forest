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
