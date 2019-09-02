import unittest
import unittest.mock
import bokeh.plotting
import os
import json
import survey


class TestSurvey(unittest.TestCase):
    def setUp(self):
        self.tool = survey.Tool()

    def test_add_root(self):
        document = bokeh.plotting.curdoc()
        document.add_root(self.tool.layout)


class TestRecords(unittest.TestCase):
    def setUp(self):
        self.path = "test-survey.json"

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_flush_writes_data_to_disk(self):
        record = {
            "timestamp": "2019-01-01 00:00:00",
            "answers": []}
        db = survey.Database(self.path)
        db.insert(record)
        db.flush()
        with open(self.path) as stream:
            contents = json.load(stream)
        result = contents["records"]
        expect = [record]
        self.assertEqual(expect, result)
