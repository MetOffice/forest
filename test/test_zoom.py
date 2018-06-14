import unittest
import forest.zoom

class TestZoom(unittest.TestCase):
    """Zoom feature"""
    def setUp(self):
        self.disjoint_boxes = [
            ((0, 0, 1, 1), (1.1, 0, 1, 1)),
            ((0, 0, 1, 1), (0, 1.1, 1, 1)),
            ((0, 0, 1, 1), (-1.1, 0, 1, 1)),
            ((0, 0, 1, 1), (0, -1.1, 1, 1)),
            ((0, 0, 1, 1), (-0.6, 0, 0.5, 1)),
            ((0, 0, 1, 1), (0, -0.6, 1, 0.5)),
        ]

    def test_is_inside_given_same_box_returns_true(self):
        self.check_is_inside((0, 0, 1, 1), (0, 0, 1, 1), True)

    def test_is_inside_given_disjoint_boxes_returns_false(self):
        for box_1, box_2 in self.disjoint_boxes:
            self.check_is_inside(box_1, box_2, False)
            self.check_is_inside(box_2, box_1, False)

    def check_is_inside(self, box_1, box_2, expect):
        result = forest.zoom.is_inside(box_1, box_2)
        try:
            self.assertEqual(result, expect)
        except AssertionError as e:
            message = "forest.zoom.is_inside({}, {}) returned {}"
            e.args += (message.format(box_1, box_2, result),)
            raise(e)
