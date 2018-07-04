import unittest
import numpy as np
import bokeh
import forest.image


class TestRGBA(unittest.TestCase):
    """Utility methods for handling RGBA data"""
    def test_get_images_given_column_data_source(self):
        column_data_source = bokeh.models.ColumnDataSource({
            "image": []
        })
        forest.image.get_images(column_data_source)

    def test_get_images_given_glyph_renderer(self):
        figure = bokeh.plotting.figure()
        glyph_renderer = figure.image_rgba(
            image=[], x=0, y=0, dw=1, dh=1
        )
        forest.image.get_images(glyph_renderer)

    def test_get_alpha_from_rgba(self):
        rgba = np.asarray([[0, 0, 0, 1], [0, 0, 0, 2]])
        result = forest.image.alpha_from_rgba(rgba)
        expect = [1, 2]
        np.testing.assert_array_almost_equal(expect, result)


class TestToggle(unittest.TestCase):
    """Controller to show/hide RGBA images"""
    def setUp(self):
        left_image = bokeh.models.ColumnDataSource()
        right_image = bokeh.models.ColumnDataSource()
        self.toggle = forest.image.Toggle(left_image, right_image)

    def test_hide_sets_alpha_to_zero(self):
        source = bokeh.models.ColumnDataSource({
            "image": [np.array([[[0, 0, 0, 2]],
                                [[0, 0, 0, 4]]])]
        })
        self.toggle.hide(source)
        result = forest.image.get_alpha(source)
        expect = [[[0],
                   [0]]]
        np.testing.assert_array_almost_equal(expect, result)

    def test_show_given_images_sets_alpha_to_original(self):
        image = bokeh.models.ColumnDataSource({
            "image": [],
        })
        self.toggle.show(image)
        self.assertIn("_alpha", image.data)
        self.assertEqual(image.data["_alpha"], [])

    def test_on_change_given_left_image_shows_left_images(self):
        self.toggle.hide = unittest.mock.Mock()
        self.toggle.show = unittest.mock.Mock()
        self.toggle.on_change("active", 1, 0)
        self.toggle.show.assert_called_once_with(self.toggle.left_images)

    def test_on_change_given_left_image_hides_right_images(self):
        self.toggle.hide = unittest.mock.Mock()
        self.toggle.show = unittest.mock.Mock()
        self.toggle.on_change("active", 1, 0)
        self.toggle.hide.assert_called_once_with(self.toggle.right_images)

    def test_on_change_given_right_image_shows_right_images(self):
        self.toggle.hide = unittest.mock.Mock()
        self.toggle.show = unittest.mock.Mock()
        self.toggle.on_change("active", 0, 1)
        self.toggle.show.assert_called_once_with(self.toggle.right_images)

    def test_on_change_given_right_image_hides_left_images(self):
        self.toggle.hide = unittest.mock.Mock()
        self.toggle.show = unittest.mock.Mock()
        self.toggle.on_change("active", 0, 1)
        self.toggle.hide.assert_called_once_with(self.toggle.left_images)
