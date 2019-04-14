import unittest
import yaml


class TestLoadConfig(unittest.TestCase):
    def test_highway_yaml(self):
        with open("highway.yaml") as stream:
            result = yaml.load(stream)
        expect = {
            "patterns": [
                {"name": "GA6", "directory": "model_data/highway_ga6*.nc"},
                {"name": "Tropical Africa 4.4km", "directory": "model_data/highway_takm4p4*.nc"},
                {"name": "East Africa 4.4km", "directory": "model_data/highway_eakm4p4*.nc"}
            ]
        }
        self.assertEqual(expect, result)
