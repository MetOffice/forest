"""
Remote procedure call driver
"""
import requests
import forest.map_view


def empty_image():
    return {
        "x": [],
        "y": [],
        "dw": [],
        "dh": [],
        "image": [],
    }


def no_args_kwargs(method):
    def inner(self, *args, **kwargs):
        return method(self)

    return inner


class Dataset:
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
    def __init__(self, url):
        self.url = f"{url}/map_view"

    @no_args_kwargs
    def image(self):
        request = requests.get(f"{self.url}/image")
        data = request.json()
        print(data)
        return data.get("result", empty_image())


class Navigator:
    def __init__(self, url):
        self.url = f"{url}/navigator"

    @no_args_kwargs
    def variables(self):
        request = requests.get(f"{self.url}/variables")
        data = request.json()
        return data.get("result", [])

    @no_args_kwargs
    def initial_times(self):
        request = requests.get(f"{self.url}/initial_times")
        data = request.json()
        return data.get("result", [])

    @no_args_kwargs
    def valid_times(self):
        request = requests.get(f"{self.url}/initial_times")
        data = request.json()
        return data.get("result", [])

    @no_args_kwargs
    def pressures(self):
        request = requests.get(f"{self.url}/pressures")
        data = request.json()
        return data.get("result", [])
