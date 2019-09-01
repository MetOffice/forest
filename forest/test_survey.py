import unittest
import bokeh.plotting
import survey


class TestSurvey(unittest.TestCase):
    def test_add_root(self):
        tool = survey.Survey()
        document = bokeh.plotting.curdoc()
        document.add_root(tool.layout)

    def test_div(self):
        tool = survey.Survey()
        result = tool.div.text
        expect = "Survey"
        self.assertEqual(expect, result)
