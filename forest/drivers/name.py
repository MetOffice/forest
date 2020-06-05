"""
NAME Atmospheric model diagnostics
"""
from forest.drivers.gridded_forecast import Dataset as _Dataset


class Dataset(_Dataset):
    def __init__(self, **kwargs):
        kwargs.update({"is_valid_cube": is_valid_cube})
        super().__init__(**kwargs)


def is_valid_cube(cube):
    """TODO: Provide a better checker"""
    return True
