import unittest
import bokeh.models
import forest.image


class TestForestImage(unittest.TestCase):
    def test_slider(self):
        left = bokeh.models.ColumnDataSource({
            "image": []
        })
        right = bokeh.models.ColumnDataSource({
            "image": []
        })
        slider = forest.image.Slider(left, right)
