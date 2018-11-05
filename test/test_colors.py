import unittest
import numpy as np
import forest.util


class TestRadarColors(unittest.TestCase):
    def test_radar_colors(self):
        result = forest.util.radar_colors()
        expect = [
            (220. / 255., 220. / 255., 220./255., 1.0),
            (122. / 255., 147. / 255., 212./255., 0.9),
            (82. / 255., 147. / 255., 212./255., 0.95),
            (39. / 255., 106. / 255., 188./255., 1.0),
            (31. / 555., 201. / 255., 26./255., 1.0),
            (253. / 255., 237. / 255., 57./255., 1.0),
            (245. / 255., 152. / 255., 0./255., 1.0),
            (235. / 255., 47. / 255., 26./255., 1.0),
            (254. / 255., 92. / 255., 252./255., 1.0),
            (255. / 255., 255. / 255., 255./255., 1.0),
        ]
        np.testing.assert_array_almost_equal(expect, result)
