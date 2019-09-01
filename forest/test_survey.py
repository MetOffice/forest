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


class TestCSV(unittest.TestCase):
    def test_constructor(self):
        csv = survey.CSV()
        csv.insert_record({})
