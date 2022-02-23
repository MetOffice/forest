"""
Xarray driver

Make it possible to use formats compatible with Xarray
"""
import xarray
import forest.map_view


def no_args_kwargs(method):
    """decorator to simplify function calls"""

    def inner(self, *args, **kwargs):
        return method(self)

    return inner


class Dataset:
    """Facade to map framework to xarray dataset"""

    def __init__(self, path, engine):
        self.xarray_dataset = xarray.open_dataset(path, engine=engine)

    def navigator(self):
        return Navigator(self.xarray_dataset)

    def map_view(self, color_mapper):
        """Construct view"""
        return forest.map_view.map_view(self.image_loader(), color_mapper)

    def image_loader(self):
        """Construct ImageLoader"""
        return ImageLoader(self.xarray_dataset)


class ImageLoader:
    """Fetch data suitable for bokeh.models.Image glyph"""

    def __init__(self, xarray_dataset):
        self.xarray_dataset = xarray_dataset

    def image(self, state):
        return self.empty_image()

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
    """Adaptor to map framework calls to Xarray API"""

    def __init__(self, xarray_dataset):
        self.xarray_dataset = xarray_dataset

    @no_args_kwargs
    def variables(self):
        return list(self.xarray_dataset.data_vars)

    @no_args_kwargs
    def initial_times(self):
        return self.xarray_dataset.forecast_reference_time.to_dict()["data"]

    @no_args_kwargs
    def valid_times(self):
        reference_time = self.xarray_dataset.forecast_reference_time[0]
        period = self.xarray_dataset.forecast_period
        return (reference_time + period).to_dict()["data"]

    @no_args_kwargs
    def pressures(self):
        return []
