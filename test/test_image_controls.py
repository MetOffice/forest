import unittest
from forest import layers


class TestControls(unittest.TestCase):
    def setUp(self):
        self.controls = layers.Controls([])

    def test_on_radio_updates_flags(self):
        cb = self.controls.on_radio(0)
        cb(None, [], [1])
        result = self.controls.flags
        expect = {
            0: [False, True, False]
        }
        self.assertEqual(expect, result)

    def test_combine(self):
        result = layers.Controls.combine(
                {0: "A", 1: "A"},
                {0: [True, False, False],
                 1: [False, False, True]})
        expect = {
            "A": [True, False, True]
        }
        self.assertEqual(expect, result)
