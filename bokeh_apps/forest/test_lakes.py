import unittest
import cartopy
import data


class TestLakes(unittest.TestCase):
    def test_10m_lakes(self):
        feature = data.at_scale(
                cartopy.feature.LAKES, "10m")
        data.xs_ys(data.iterlines(
            feature.geometries()))
