import unittest
import datetime as dt
import os
import glob
import rdt


class TestRDT(unittest.TestCase):
    def setUp(self):
        self.date = dt.datetime(2019, 4, 17)
        self.paths = glob.glob(os.path.expanduser("~/cache/RDT_features_eastafrica_*.json"))
        self.loader = rdt.Loader(self.paths)

    @unittest.skip("implementing simple test")
    def test_view(self):
        view = rdt.View(self.loader)
        view.render(self.date)

    @unittest.skip("implementing simple test")
    def test_loader(self):
        geojson = self.loader.load_date(self.date)
        print(geojson)
        self.assertTrue(False)

    def test_find_file(self):
        date = dt.datetime(2019, 3, 15, 12, 0)
        result = self.loader.find_file(date)
        expect = "/Users/andrewryan/cache/RDT_features_eastafrica_201903151200.json"
        self.assertEqual(expect, result)

    def test_parse_date(self):
        path = "/Users/andrewryan/cache/RDT_features_eastafrica_201903151215.json"
        result = self.loader.parse_date(path)
        expect = dt.datetime(2019, 3, 15, 12, 15)
        self.assertEqual(expect, result)
