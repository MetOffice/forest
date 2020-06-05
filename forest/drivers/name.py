"""
NAME Atmospheric model diagnostics
"""
import datetime as dt
from forest.drivers.gridded_forecast import Dataset as _Dataset
from forest.drivers.gridded_forecast import Navigator as _Navigator


class Dataset(_Dataset):
    def __init__(self, **kwargs):
        kwargs.update({"is_valid_cube": is_valid_cube})
        super().__init__(**kwargs)

    def navigator(self):
        return Navigator(self._paths, self.is_valid_cube)


class Navigator(_Navigator):
    def initial_times(self, pattern, variable):
        return [dt.datetime(1970, 1, 1)]


def is_valid_cube(cube):
    """TODO: Provide a better checker"""
    return True
