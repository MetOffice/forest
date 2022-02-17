"""
Remote procedure call driver

.. note: See example in `forest/apps/rpc-server`
"""
import requests
import forest.map_view


def no_args_kwargs(method):
    """decorator to simplify function calls"""

    def inner(self, *args, **kwargs):
        return method(self)

    return inner


class Dataset:
    """Remote procedure call dataset"""

    def __init__(self, url):
        self.url = url

    def navigator(self):
        return Navigator(self.url)

    def map_view(self, color_mapper):
        """Construct view"""
        return forest.map_view.map_view(self.image_loader(), color_mapper)

    def image_loader(self):
        """Construct ImageLoader"""
        return ImageLoader(self.url)


class ImageLoader:
    """Fetch data suitable for bokeh.models.Image glyph"""

    def __init__(self, url):
        self.root = f"{url}/map_view"

    def image(self, state):
        request = requests.get(
            f"{self.root}/image",
            params={
                "valid_time": state.valid_time,
                "initial_time": state.initial_time,
                "pressure": state.pressure,
                "variable": state.variable,
            },
        )
        data = request.json()
        return data.get("result", self.empty_image())

    @staticmethod
    def empty_image():
        return {
            "x": [],
            "y": [],
            "dw": [],
            "dh": [],
            "image": [],
        }


class Navigator:
    """Adaptor to map framework calls to RPC fetch requests"""

    def __init__(self, url):
        self.root = f"{url}/navigator"

    @no_args_kwargs
    def variables(self):
        return self.fetch(f"{self.root}/variables")

    @no_args_kwargs
    def initial_times(self):
        return self.fetch(f"{self.root}/initial_times")

    @no_args_kwargs
    def valid_times(self):
        return self.fetch(f"{self.root}/valid_times")

    @no_args_kwargs
    def pressures(self):
        return self.fetch(f"{self.root}/pressures")

    @staticmethod
    def fetch(endpoint):
        request = requests.get(endpoint)
        data = request.json()
        return data.get("result", [])
