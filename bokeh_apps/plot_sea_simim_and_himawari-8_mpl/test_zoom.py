import unittest
import zoom

class TestZoom(unittest.TestCase):
    """Zoom feature"""
    def test_overlap(self):
        self.check_overlap((0, 0, 1, 1), (0, 0, 1, 1), True)

    def test_overlap_given_disjoint_in_x(self):
        self.check_overlap((0, 0, 1, 1), (2, 0, 1, 1), False)

    def test_overlap_given_disjoint_in_y(self):
        self.check_overlap((0, 0, 1, 1), (0, 2, 1, 1), False)

    def test_overlap_given_overlapping_with_different_x(self):
        self.check_overlap((0, 0, 1, 1), (0.5, 0, 1, 1), True)

    def test_overlap_given_different_y_returns_true(self):
        self.check_overlap((0, 0, 1, 1), (0, 0.5, 1, 1), True)

    def test_overlap_given_small_box_to_left_returns_false(self):
        self.check_overlap((0, 0, 1, 1), (-0.6, 0, 0.5, 1), False)

    def test_overlap_given_small_box_below_returns_false(self):
        self.check_overlap((0, 0, 1, 1), (0, -0.6, 1, 0.5), False)

    def check_overlap(self, box_1, box_2, expect):
        result = zoom.boxes_overlap(box_1, box_2)
        self.assertEqual(result, expect)
