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


class TestReducer(unittest.TestCase):
    def test_reducer_given_show_results(self):
        action = survey.show_results()
        result = survey.reducer({}, action)
        expect = {
            "page": survey.RESULTS
        }
        self.assertEqual(expect, result)

    def test_reducer_given_show_welcome(self):
        action = survey.show_welcome()
        result = survey.reducer({}, action)
        expect = {
            "page": survey.WELCOME
        }
        self.assertEqual(expect, result)

    def test_reducer_given_reset_answers(self):
        action = survey.reset_answers()
        initial = {"answers": ["y", "text"]}
        state = survey.reducer(initial, action)
        result = state["answers"]
        expect = []
        self.assertEqual(expect, result)


class TestResultsPage(unittest.TestCase):
    def setUp(self):
        self.results = survey.Results()

    def test_on_welcome(self):
        listener = unittest.mock.Mock()
        self.results.subscribe(listener)
        self.results.on_welcome()
        expect = survey.show_welcome()
        listener.assert_called_once_with(expect)

    def test_render_given_state(self):
        state = {
            "results": [
                {
                    "timestamp": "2019-01-01 00:00:00",
                    "answers": [
                        "y", "Some text"
                    ]
                }
            ]
        }
        self.results.render(state)
        result = self.results.bar_source.data
        expect = {
            "x": ["Yes", "No"],
            "top": [1, 0]
        }
        self.assertEqual(expect, result)

    def test_render_populates_data_table(self):
        state = {
            "results": [
                {
                    "timestamp": "2019-01-01 00:00:00",
                    "answers": [
                        "y", "Some text"
                    ]
                }
            ]
        }
        self.results.render(state)
        result = self.results.table_source.data
        expect = {
            "timestamp": ["2019-01-01 00:00:00"],
            "answer": ["Some text"]
        }
        self.assertEqual(expect, result)


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
