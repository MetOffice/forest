from unittest.mock import Mock, sentinel
import forest.gallery


def test_gallery():
    store = Mock()
    state = {}
    factory_class = Mock()
    context = Mock()
    gallery = forest.gallery.Gallery.series_view({"label": sentinel.dataset},
                                                 sentinel.figure)
    gallery.connect(store)
    gallery.render(state)
