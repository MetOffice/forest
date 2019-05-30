import unittest
import yaml
import app.main


class TestMain(unittest.TestCase):
    def test_main_loads_config(self):
        content = yaml.dump({
            "models": [
                {"name": "Label",
                 "pattern": "*.nc"}
            ]
        })
        config = app.main.load_config(content)
        result = config.patterns
        expect = [("Label", "*.nc")]
        self.assertEqual(expect, result)
