import unittest
import datetime as dt
import glob
from forest import earth_networks
import pandas as pd
import pandas.testing as pdt


@unittest.skip("green light")
class TestEarthNetworks(unittest.TestCase):
    def setUp(self):
        self.pattern = "/Users/andrewryan/cache/engl*.txt"
        self.paths = glob.glob(self.pattern)

    def test_load_date(self):
        fixture = earth_networks.Loader(self.paths)
        frame = fixture.load_date(dt.datetime(2019, 4, 17))
        result = frame.iloc[0]
        expect = pd.Series([
            dt.datetime(2019, 4, 17, 0, 0, 1, 440000),
            "IC",
            2.75144,
            31.92064], index=[
                "date",
                "flash_type",
                "latitude",
                "longitude"
            ],
            name=0)
        pdt.assert_series_equal(expect, result)
