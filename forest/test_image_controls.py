import unittest
import images


class TestControls(unittest.TestCase):
    def setUp(self):
        self.controls = images.Controls([])

    def test_on_dropdown_updates_models(self):
        cb = self.controls.on_dropdown(0)
        cb(None, None, "GA6")
        result = self.controls.models
        expect = {
            0: "GA6"
        }
        self.assertEqual(expect, result)

    def test_on_radio_updates_flags(self):
        cb = self.controls.on_radio(0)
        cb(None, [], [1])
        result = self.controls.flags
        expect = {
            0: [False, True, False]
        }
        self.assertEqual(expect, result)

    def test_remove_row_removes_state(self):
        self.controls.add_row()
        self.controls.on_dropdown(1)(None, None, "key")
        self.controls.on_radio(1)(None, [], [1])
        self.controls.remove_row()
        self.assertEqual(self.controls.models, {})
        self.assertEqual(self.controls.flags, {})

    def test_combine(self):
        result = images.Controls.combine(
                {0: "A", 1: "A"},
                {0: [True, False, False],
                 1: [False, False, True]})
        expect = {
            "A": [True, False, True]
        }
        self.assertEqual(expect, result)
