"""
NAME Atmospheric model diagnostics
"""
from functools import partial
import iris
import datetime as dt
import forest.util
from forest.drivers.gridded_forecast import Dataset as _Dataset
from forest.drivers.gridded_forecast import Navigator as _Navigator
from forest.drivers.gridded_forecast import ImageLoader as _ImageLoader


class Dataset(_Dataset):
    def __init__(self, **kwargs):
        super().__init__(is_valid_cube=is_valid_cube,
                         image_loader_class=ImageLoader,
                         **kwargs)

    def navigator(self):
        return Navigator(self._paths, self.is_valid_cube)


class Navigator(_Navigator):
    def initial_times(self, pattern, variable):
        return [dt.datetime(1970, 1, 1)]

    def valid_times(self, pattern, variable, initial_time):
        return [forest.util.to_datetime(t)
                for t in super().valid_times(pattern, variable, initial_time)]


class ImageLoader(_ImageLoader):
    @staticmethod
    def select_image(cube, valid_datetime):
        def func(valid_datetime, cell):
            print(cell.bound, valid_datetime)
            return cell.bound[0] < valid_datetime <= cell.bound[1]
        return cube.extract(iris.Constraint(time=partial(func, valid_datetime)))


def is_valid_cube(cube):
    """TODO: Provide a better checker"""
    return True
