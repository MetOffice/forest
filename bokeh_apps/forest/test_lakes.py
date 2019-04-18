import unittest
import cartopy
import data


class TestLakes(unittest.TestCase):
    def test_10m_lakes(self):
        data.feature_lines(data.at_scale(cartopy.feature.LAKES, "10m"))
