"""GPM driver"""
from functools import partial, lru_cache
import glob
import datetime as dt
import netCDF4
import numpy as np
import forest.view
import forest.geo
import forest.util


@lru_cache(maxsize=16)
def read_times(path):
    """Read time axis from a file"""
    with netCDF4.Dataset(path) as dataset:
        var = dataset.variables["time"]
        times = netCDF4.num2date(var[:], units=var.units)
    return np.array([forest.util.to_datetime(t) for t in times], dtype=object)


class Dataset:
    def __init__(self, pattern=None, **kwargs):
        self.pattern = pattern
        self.locator = Locator()

    def navigator(self):
        return Navigator(self.pattern, self.locator)

    def map_view(self, color_mapper):
        return forest.view.UMView(_Loader(self.pattern,
                                          self.locator),
                                  color_mapper,
                                  use_hover_tool=False)


class Locator:
    """Search files to find paths"""
    def __init__(self):
        self.parse_date = partial(forest.util.parse_date,
                                  "[0-9]{8}", "%Y%m%d")

    def find_paths_and_index(self, paths, date):
        """Flatten paths and index generators"""
        window_size = dt.timedelta(days=1)
        for path in self.find_paths(paths, date, window_size):
            for index in self.find_index(path, date):
                yield path, index

    def find_paths(self, paths, search_date, window_size):
        for path in paths:
            if abs(self.parse_date(path) - search_date) < window_size:
                yield path

    def find_index(self, path, date):
        times = read_times(path)
        tolerance = dt.timedelta(minutes=1)
        pts = np.where(np.abs(times - date) < tolerance)
        for index in pts[0]:
            yield index


class Navigator:
    def __init__(self, pattern, locator):
        self.pattern = pattern
        self.locator = locator
        self._time_arrays = {}

    def variables(self, *args, **kwargs):
        return ["precipitation_flux"]

    def initial_times(self, *args, **kwargs):
        return [dt.datetime(1970, 1, 1)]

    def valid_times(self, *args, valid_times=None, valid_time=None, **kwargs):
        """Times from time stamps and contents of file(s)"""
        paths = list(sorted(glob.glob(self.pattern)))
        timestamps = {self.locator.parse_date(path): path for path in paths}

        # Guard clause uninitialised state
        if (valid_times is None) or (valid_time is None):
            return list(timestamps.keys())

        # Cache path time axis
        if valid_time in timestamps:
            path = timestamps[valid_time]
            self._time_arrays[valid_time] = read_times(path)

        # Compute dataset time axis
        arrays = [
            np.asarray(list(timestamps.keys()))
        ]
        for array in self._time_arrays.values():
            arrays.append(array)
        if len(arrays) == 0:
            return []
        return np.unique(np.concatenate(arrays))

    def pressures(self, *args, **kwargs):
        return []


class _Loader:
    """Compatible with forest.view.UMView"""
    def __init__(self, pattern, locator):
        self.pattern = pattern
        self.locator = locator
        self.empty_image = {
            "x": [],
            "y": [],
            "dw": [],
            "dh": [],
            "image": [],
        }

    def image(self, old_state):
        if old_state.valid_time is None:
            return self.empty_image

        # Default value
        data = self.empty_image

        # Search file system
        paths = sorted(glob.glob(self.pattern))
        date = old_state.valid_time
        for path, index in self.locator.find_paths_and_index(paths, date):
            with netCDF4.Dataset(path) as dataset:
                lons = dataset.variables["longitude"][:]
                lats = dataset.variables["latitude"][:]
                data = dataset.variables["precipitation_flux"][index]
            npixels = 512
            data = forest.geo.stretch_image(lons, lats, data,
                                    plot_height=npixels,
                                    plot_width=npixels)
            break
        return data
