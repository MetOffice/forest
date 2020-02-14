"""
Read GHRSST L4 data.

These conventions should conform the GDS 2.0 conventions (based on cf1.7)
See:
https://www.ghrsst.org/about-ghrsst/governance-documents/

"""

from datetime import datetime
import collections

import numpy as np
try:
    import iris
except ModuleNotFoundError:
    # ReadTheDocs can't import iris
    iris = None

from forest import geo


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


def _to_datetime(d):
    if isinstance(d, datetime):
        return d
    elif isinstance(d, str):
        try:
            return datetime.strptime(d, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return datetime.strptime(d, "%Y-%m-%dT%H:%M:%S")
    elif isinstance(d, np.datetime64):
        return d.astype(datetime)
    else:
        raise Exception("Unknown value: {}".format(d))


def coordinates(valid_time, initial_time, pressures, pressure):
    valid = _to_datetime(valid_time)
    initial = _to_datetime(initial_time)
    hours = (valid - initial).total_seconds() / (60*60)
    length = "T{:+}".format(int(hours))
    level = "Sea Surface"
    return {
        'valid': [valid],
        'initial': [initial],
        'length': [length],
        'level': [level]
    }


def _is_valid_cube(cube):
    """Return True if, and only if, the cube conforms to a GHRSST data specification"""
    attributes = cube.metadata.attributes
    is_gds = ("GDS_version_id" in attributes) or ("gds_version_id" in attributes)
    dim_names = [c.name() for c in cube.dim_coords]
    contains_dims = {'time', 'latitude', 'longitude'}.issubset(set(dim_names))
    dims_are_ordered = dim_names[:3] == ['time', 'latitude', 'longitude']
    has_3_dims = len(dim_names) == 3
    return is_gds and contains_dims and dims_are_ordered and has_3_dims

# TODO: This logic should move to a "Group" concept.
def _load(pattern):
    """Return all the valid GHRSST L4 cubes that can be loaded
    from the given filename pattern."""
    cubes = iris.load(pattern)

    # Ensure that we only retain cubes that meet our entry criteria
    # for "gridded forecast"
    cubes = list(filter(_is_valid_cube, cubes))
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
    key_func = lambda t: "0" if t[0] == "sea_surface_foundation_temperature" else t[0]
    cube_mapping_ordered = collections.OrderedDict(
        sorted(cube_mapping.items(), key=key_func))
    return cube_mapping_ordered


class ImageLoader:
    def __init__(self, label, pattern):
        self._label = label
        self._cubes = _load(pattern)

    def image(self, state):
        cube = self._cubes[state.variable]
        valid_datetime = _to_datetime(state.valid_time)
        cube = cube.extract(iris.Constraint(time=valid_datetime))

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


class Navigator:
    def __init__(self, paths):
        self._cubes = _load(paths)

    def variables(self, pattern):
        return list(self._cubes.keys())

    def initial_times(self, pattern, variable=None):
        return list([datetime(1970,1,1)])

    def valid_times(self, pattern, variable, initial_time):
        cube = self._cubes[variable]
        return [cell.point for cell in cube.coord('time').cells()]

    def pressures(self, pattern, variable, initial_time):
        pressures = []
        return pressures
