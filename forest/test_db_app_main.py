import unittest
import yaml
import forest.app.main as main


class TestMain(unittest.TestCase):
    def test_main_loads_config(self):
        content = yaml.dump({
            "models": [
                {"name": "Label",
                 "pattern": "*.nc"}
            ]
        })
        config = main.load_config(content)
        result = config.patterns
        expect = [("Label", "*.nc")]
        self.assertEqual(expect, result)
