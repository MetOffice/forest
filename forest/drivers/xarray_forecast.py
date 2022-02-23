"""
Xarray driver

Make it possible to use formats compatible with Xarray
"""
import numpy as np
import xarray
import forest.map_view
import forest.geo
from forest.util import to_datetime


def no_args_kwargs(method):
    """decorator to simplify function calls"""

    def inner(self, *args, **kwargs):
        return method(self)

    return inner


class Dataset:
    """Facade to map framework to xarray dataset"""

    def __init__(self, path, engine, label=""):
        self.label = label
        self.xarray_dataset = xarray.open_dataset(path, engine=engine)

    def navigator(self):
        return Navigator(self.xarray_dataset)

    def map_view(self, color_mapper):
        """Construct view"""
        tooltips = [
            ("Name", "@name"),
            ("Variable", "@variable"),
            ("Value", "@image @units"),
            ("Level", "@level"),
            ("Valid at", "@valid{%F %H:%M}"),
            ("Forecast reference", "@initial{%F %H:%M}"),
            ("Forecast length", "@length"),
        ]
        return forest.map_view.map_view(self.image_loader(), color_mapper,
                                        tooltips=tooltips)

    def image_loader(self):
        """Construct ImageLoader"""
        return ImageLoader(self.xarray_dataset, label=self.label)


class ImageLoader:
    """Fetch data suitable for bokeh.models.Image glyph"""

    def __init__(self, xarray_dataset, label=""):
        self.xarray_dataset = xarray_dataset
        self.label = label

    def image(self, state):
        forecast_reference_time = to_datetime(state.initial_time)
        forecast_period = to_datetime(state.valid_time) - forecast_reference_time
        data_array = self.xarray_dataset[state.variable]
        try:
            data_array = data_array.sel(
                forecast_reference_time=forecast_reference_time,
                forecast_period=forecast_period,
            )
            print(data_array)
        except KeyError as error:
            print(f"Caught KeyError: {error}")
            return self.empty()
        x = np.ma.masked_invalid(data_array.longitude)[:]
        y = np.ma.masked_invalid(data_array.latitude)[:]
        z = np.ma.masked_invalid(data_array)[:]
        if z.mask.all():
            return self.empty()
        else:
            bokeh_data = forest.geo.stretch_image(x, y, z)
            bokeh_data.update(
                self.metadata(
                    self.label,
                    state.variable,
                    getattr(data_array, "units", ""),
                    forecast_reference_time,
                    forecast_period,
                )
            )
            print(bokeh_data)
            return bokeh_data

    @staticmethod
    def metadata(name, variable, units, forecast_reference_time, forecast_period):
        hours = (forecast_period).total_seconds() / (60 * 60)
        length = "T{:+}".format(int(hours))
        return {
            "name": [name],
            "variable": [variable],
            "units": [units],
            "valid": [forecast_reference_time + forecast_period],
            "initial": [forecast_reference_time],
            "length": [length],
            "level": [1.5],
        }

    def empty(self):
        """Default bokeh column data source structure"""
        image = self.empty_image()
        image.update(self.empty_metadata())
        return image

    @staticmethod
    def empty_image():
        return {
            "x": [],
            "y": [],
            "dw": [],
            "dh": [],
            "image": [],
        }

    @staticmethod
    def empty_metadata():
        return {
            "length": [],
            "initial": [],
            "valid": [],
            "level": [],
            "units": [],
            "variable": [],
            "name": [],
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

    def valid_times(self, pattern, variable, initial_time):
        """List of validity times from a particular model run"""
        forecast_reference_time = self.xarray_dataset.forecast_reference_time.sel(
            forecast_reference_time=initial_time
        )
        forecast_period = self.xarray_dataset.forecast_period
        return (forecast_reference_time + forecast_period).to_dict()["data"]

    @no_args_kwargs
    def pressures(self):
        return []
