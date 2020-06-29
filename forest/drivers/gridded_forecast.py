from datetime import datetime
import collections

import numpy as np
try:
    import iris
except ModuleNotFoundError:
    # ReadTheDocs can't import iris
    iris = None

import cftime

import glob
from forest import geo
import forest.map_view
from forest.util import to_datetime as _to_datetime


def empty_image():
    return {
        "x": [],
        "y": [],
        "dw": [],
        "dh": [],
        "image": [],
        "name": [],
        "units": [],
        "valid": [],
        "initial": [],
        "length": [],
        "level": []
    }


def coordinates(valid_time, initial_time, pressures, pressure):
    valid = _to_datetime(valid_time)
    initial = _to_datetime(initial_time)
    hours = (valid - initial).total_seconds() / (60*60)
    length = "T{:+}".format(int(hours))
    if (len(pressures) > 0) and (pressure is not None):
        level = "{} hPa".format(int(pressure))
    else:
        level = "Surface"
    return {
        'valid': [valid],
        'initial': [initial],
        'length': [length],
        'level': [level]
    }


def _is_valid_cube(cube):
    """Return True if, and only if, the cube meets our criteria for a
    'gridded forecast'."""
    dim_names = [coord.name() for coord in cube.dim_coords]
    return (2 <= cube.ndim <= 3
            and len(cube.dim_coords) == cube.ndim
            and (dim_names == ['time', 'latitude', 'longitude'] or
                 (dim_names == ['latitude', 'longitude'] and
                  len(cube.coords('time')) == 1))
            and len(cube.coords('forecast_reference_time')) == 1)


# TODO: This logic should move to a "Group" concept.
def _load(pattern, is_valid_cube=None):
    """Return all the valid gridded forecast cubes that can be loaded
    from the given filename pattern."""
    if is_valid_cube is None:
        is_valid_cube = _is_valid_cube

    cubes = iris.load(pattern)

    # Ensure that we only retain cubes that meet our entry criteria
    # for "gridded forecast"
    cubes = list(filter(is_valid_cube, cubes))
    assert len(cubes) > 0

    # Find all the names with duplicates
    name_counts = collections.Counter(cube.name() for cube in cubes)
    duplicate_names = {name for name, count in name_counts.items()
                       if count > 1}

    # Map names (with numeric suffixes for duplicates) to cubes
    duplicate_counts = collections.defaultdict(int)
    cube_mapping = {}
    for cube in cubes:
        name = cube.name()
        if name in duplicate_names:
            duplicate_counts[name] += 1
            name += f' ({duplicate_counts[name]})'
        cube_mapping[name] = cube
    return cube_mapping


class Dataset:
    """High-level class to relate navigators, loaders and views"""
    def __init__(self, label=None, pattern=None, **kwargs):
        self._label = label
        self.pattern = pattern
        if pattern is not None:
            self._paths = glob.glob(pattern)
        else:
            self._paths = []

    def navigator(self):
        """Construct navigator"""
        cube_dict = _load(self._paths, _is_valid_cube)
        return Navigator(cube_dict)

    def map_view(self, color_mapper):
        """Construct view"""
        return forest.map_view.map_view(self.image_loader(), color_mapper)

    def image_loader(self):
        """Construct ImageLoader"""
        cube_dict = _load(self._paths, _is_valid_cube)
        return ImageLoader(self._label, cube_dict)


class ImageLoader:
    def __init__(self, label, cube_dict,
                 extract_cube=None):
        self._label = label
        self._cubes = cube_dict
        if extract_cube is not None:
            self.extract_cube = extract_cube

    def image(self, state):
        cube = self._cubes[state.variable]
        valid_datetime = _to_datetime(state.valid_time)
        cube = self.extract_cube(cube, valid_datetime)
        if cube is None:
            data = empty_image()
        else:
            data = geo.stretch_image(cube.coord('longitude').points,
                                     cube.coord('latitude').points, cube.data)
            data.update(coordinates(state.valid_time, state.initial_time,
                                    state.pressures, state.pressure))
            data.update({
                'name': [self._label],
                'units': [str(cube.units)]
            })
        return data

    @staticmethod
    def extract_cube(cube, valid_datetime):
        """Extract 2D image slice from cube"""
        return cube.extract(iris.Constraint(time=valid_datetime))


class Navigator:
    def __init__(self, cube_dict):
        self._cubes = cube_dict

    def variables(self, pattern):
        return list(self._cubes.keys())

    def initial_times(self, pattern, variable=None):
        return list({cube.coord('forecast_reference_time').cell(0).point
                     for cube in self._cubes.values()})

    def valid_times(self, pattern, variable, initial_time):
        cube = self._cubes[variable]
        return [cell.point for cell in cube.coord('time').cells()]

    def pressures(self, pattern, variable, initial_time):
        cube = self._cubes[variable]
        pressures = []
        try:
            pressures = [cell.point for cell in cube.coord('pressure').cells()]
        except iris.exceptions.CoordinateNotFoundError:
            pass
        return pressures
