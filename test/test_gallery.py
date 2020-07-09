from unittest.mock import Mock
import forest.gallery


def test_gallery():
    store = Mock()
    scaling_group = Mock()
    gallery = forest.gallery.Gallery({"label": scaling_group})
    gallery.connect(store)
