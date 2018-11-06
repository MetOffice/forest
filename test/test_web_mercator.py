import numpy as np
import unittest
import cartopy
import forest

class TestWebMercator(unittest.TestCase):
    def test_web_mercator(self):
        result = forest.web_mercator(0, 0)
        expect = (0, 0)
        self.assert_coordinates_equal(expect, result)

    def test_web_mercator_given_realistic_values(self):
        result = forest.web_mercator(0, 45)
        expect = (0, 5621521.48619207)
        self.assert_coordinates_equal(expect, result)

    def assert_coordinates_equal(self, expect, result):
        ex, ey = expect
        rx, ry = result
        self.assertEqual(np.shape(ex), rx.shape)
        self.assertEqual(np.shape(ey), ry.shape)
        np.testing.assert_array_almost_equal(ex, rx)
        np.testing.assert_array_almost_equal(ey, ry)

class TestTransform(unittest.TestCase):
    def setUp(self):
        self.plate_carree = cartopy.crs.PlateCarree()
        self.google = cartopy.crs.Mercator.GOOGLE

    def test_transform_to_from_should_return_original_values(self):
        lons = np.linspace(-180, 180, 100)
        lats = np.linspace(-85, 85, 100)
        source = self.plate_carree
        target = self.google
        x, y = forest.transform(source, target, lons, lats)
        trans_lons, trans_lats = forest.transform(target, source, x, y)
        np.testing.assert_array_almost_equal(trans_lons, lons)
        np.testing.assert_array_almost_equal(trans_lats, lats)

    def test_unstable_transform(self):
        xy_extent = -14025626.369129315, 31948064.386846364, -6594114.510220122, 16416534.01834311
        print(xy_extent)
        x, y = unpack(xy_extent)
        lons, lats = forest.transform(self.google, self.plate_carree, x, y)
        print(lons, lats)
        x, y = forest.transform(self.plate_carree, self.google, lons, lats)
        print(x, y)
        lons, lats = forest.transform(self.google, self.plate_carree, x, y)
        print(lons, lats)
        self.assertTrue(False)

def unpack(extent):
    return [extent[0], extent[1]], [extent[2], extent[3]]
