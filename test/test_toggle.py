import unittest
import forest.image


class TestToggle(unittest.TestCase):
    def test_constructor(self):
        left_image, right_image = None, None
        toggle = forest.image.Toggle(left_image, right_image)
