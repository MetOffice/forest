import unittest
import unittest.mock
import bokeh.plotting
import survey


class TestSurvey(unittest.TestCase):
    def setUp(self):
        self.tool = survey.Tool()

    def test_add_root(self):
        document = bokeh.plotting.curdoc()
        document.add_root(self.tool.layout)

    def test_div(self):
        result = self.tool.div.text
        expect = "Survey"
        self.assertEqual(expect, result)

    def test_submit_button(self):
        result = self.tool.buttons["submit"]
        expect = bokeh.models.Button
        self.assertIsInstance(result, expect)

    def test_on_save_emits_action(self):
        listener = unittest.mock.Mock()
        self.tool.subscribe(listener)
        self.tool.on_save()
        expect = {
            "kind": "SUBMIT",
            "payload": {}
        }
        listener.assert_called_once_with(expect)


class TestCSV(unittest.TestCase):
    def test_constructor(self):
        csv = survey.CSV()
        csv.insert_record({})
