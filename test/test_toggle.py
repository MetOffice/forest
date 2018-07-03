import unittest
import bokeh
import forest.image


class TestToggle(unittest.TestCase):
    """Controller to show/hide RGBA images"""
    def setUp(self):
        left_image = bokeh.models.ColumnDataSource()
        right_image = bokeh.models.ColumnDataSource()
        self.toggle = forest.image.Toggle(left_image, right_image)

    def test_hide_given_images_sets_alpha_to_zero(self):
        image = bokeh.models.ColumnDataSource()
        self.toggle.hide(image)

    def test_show_given_images_sets_alpha_to_original(self):
        image = bokeh.models.ColumnDataSource({
            "image": [],
        })
        self.toggle.show(image)
        self.assertIn("_alpha", image.data)

    def test_on_change_given_left_image_shows_left_images(self):
        self.toggle.show = unittest.mock.Mock()
        self.toggle.on_change("active", 1, 0)
        self.toggle.show.assert_called_once_with(self.toggle.left_images)

    def test_on_change_given_left_image_hides_right_images(self):
        self.toggle.hide = unittest.mock.Mock()
        self.toggle.on_change("active", 1, 0)
        self.toggle.hide.assert_called_once_with(self.toggle.right_images)

    def test_on_change_given_right_image_shows_right_images(self):
        self.toggle.show = unittest.mock.Mock()
        self.toggle.on_change("active", 0, 1)
        self.toggle.show.assert_called_once_with(self.toggle.right_images)

    def test_on_change_given_right_image_hides_left_images(self):
        self.toggle.hide = unittest.mock.Mock()
        self.toggle.on_change("active", 0, 1)
        self.toggle.hide.assert_called_once_with(self.toggle.left_images)
