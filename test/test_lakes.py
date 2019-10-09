import unittest
import cartopy
from forest import data


class TestLakes(unittest.TestCase):
    def test_10m_lakes(self):
        feature = cartopy.feature.NaturalEarthFeature(
            'physical', 'lakes', '10m')
        data.xs_ys(data.iterlines(
            feature.geometries()))


class TestLoadCoastlines(unittest.TestCase):
    def test_load_coastlines(self):
        data.load_coastlines()
