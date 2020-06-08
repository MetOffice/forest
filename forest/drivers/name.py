"""
NAME Atmospheric model diagnostics
"""
from functools import partial
import iris
import datetime as dt
import forest.util
from forest.drivers.gridded_forecast import _load
from forest.drivers.gridded_forecast import Dataset as _Dataset
from forest.drivers.gridded_forecast import Navigator as _Navigator
from forest.drivers.gridded_forecast import ImageLoader


class Dataset(_Dataset):
    """Provide dataset specific functionality"""
    def navigator(self):
        """Construct a Navigator"""
        cube_dict = _load(self._paths, is_valid_cube)
        return Navigator(cube_dict)

    def image_loader(self):
        cube_dict = _load(self._paths, is_valid_cube)
        return ImageLoader(self._label, cube_dict,
                           extract_cube=extract_cube)


class Navigator(_Navigator):
    """Navigate dataset dimensions"""
    def initial_times(self, pattern, variable):
        """Forecast initialisation time"""
        return [dt.datetime(1970, 1, 1)]

    def valid_times(self, pattern, variable, initial_time):
        """Time(s) for which fields are valid"""
        return [forest.util.to_datetime(t)
                for t in super().valid_times(pattern, variable, initial_time)]


def extract_cube(cube, valid_datetime):
    """Select a single 2D slice related to validity time"""
    def func(valid_datetime, cell):
        return cell.bound[0] < valid_datetime <= cell.bound[1]
    return cube.extract(iris.Constraint(time=partial(func, valid_datetime)))


def is_valid_cube(cube):
    """TODO: Provide a better checker"""
    return True
