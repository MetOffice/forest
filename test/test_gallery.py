from unittest.mock import Mock, sentinel
import forest.gallery


def test_gallery():
    store = Mock()
    state = {}
    factory_class = Mock()
    context = Mock()
    gallery = forest.gallery.Gallery._from_dataset({"label": sentinel.dataset},
                                                   factory_class)
    gallery.connect(store)
    gallery.render(state)
    factory_class.assert_called_once_with(sentinel.dataset)
