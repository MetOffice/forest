import unittest
import datetime as dt
import os
import glob
import json
import rdt


class TestRDT(unittest.TestCase):
    def setUp(self):
        pattern = os.path.join(os.path.dirname(__file__), "sample/RDT*.json")
        self.loader = rdt.Loader(pattern)

    def test_paths(self):
        result = [os.path.basename(path) for path in self.loader.paths]
        expect = ["RDT_features_eastafrica_201904171245.json"]
        self.assertEqual(expect, result)

    @unittest.skip("cache changed")
    def test_view(self):
        date = dt.datetime(2019, 4, 17)
        view = rdt.View(self.loader)
        view.render(date)
        data = json.loads(view.source.geojson)
        result = data['features'][0]['geometry']['coordinates'][0][0]
        expect = [4313964.2267117305, 739588.7725023176]
        self.assertEqual(expect, result)

    @unittest.skip("cache changed")
    def test_loader(self):
        date = dt.datetime(2019, 4, 17)
        geojson = self.loader.load_date(date)
        result = json.loads(geojson)['features'][0]['geometry']['coordinates'][0][0]
        expect = [4313964.2267117305, 739588.7725023176]
        self.assertEqual(expect, result)

    def test_find_file(self):
        date = dt.datetime(2019, 3, 15, 12, 0)
        result = os.path.basename(self.loader.find_file(date))
        expect = "RDT_features_eastafrica_201904171245.json"
        self.assertEqual(expect, result)

    def test_parse_date(self):
        path = "/Users/andrewryan/cache/RDT_features_eastafrica_201903151215.json"
        result = self.loader.parse_date(path)
        expect = dt.datetime(2019, 3, 15, 12, 15)
        self.assertEqual(expect, result)
