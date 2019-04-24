import unittest
import yaml


def as_patterns(data):
    d = {}
    for row in data["patterns"]:
        d[row['name']] = row['directory']
    return d


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

    def test_offline(self):
        with open("highway-offline.yaml") as stream:
            data = yaml.load(stream)
        result = as_patterns(data)
        expect = {
            "OS42": "highway_os42_ea_*.nc",
            "EIDA50": "EIDA50_takm4p4_*.nc",
            "RDT": "*.json"
        }
        self.assertEqual(expect, result)
