import unittest
import yaml
import os
import forest


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.path = "test-config.yaml"

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_load_config(self):
        data = {
            "files": [
                {"name": "EIDA50",
                 "directory": "~/cache",
                 "pattern": "*.nc"}
            ]
        }
        with open(self.path, "w") as stream:
            yaml.dump(data, stream)
        result = forest.load_config(self.path).data
        expect = data
        self.assertEqual(expect, result)

    def test_patterns(self):
        data = {
            "files": []
        }
        with open(self.path, "w") as stream:
            yaml.dump(data, stream)
        config = forest.load_config(self.path)
        result = config.patterns
        expect = []
        self.assertEqual(expect, result)

    def test_patterns(self):
        config = forest.config.Config({
            "files": [
                {
                    "name": "Name",
                    "pattern": "*.nc"
                }
            ]
        })
        result = config.patterns
        expect = [("Name", "*.nc")]
        self.assertEqual(expect, result)
