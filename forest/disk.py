"""Helpers to locate data on disk"""
import netCDF4
import datetime as dt
import numpy as np
from forest import util
import fnmatch
import iris
from forest.exceptions import (
        InitialTimeNotFound,
        PressuresNotFound,
        ValidTimesNotFound)
from forest.db.exceptions import SearchFail


class AxisNotFound(Exception):
    pass


def ndindex(masks, axes):
    """N-dimensional array indexing

    Given logical masks and their axes generate
    a multi-dimensional slice

    :returns: tuple(slices)
    """
    joint = {}
    for mask, axis in zip(masks, axes):
        print(mask, axis)
        if axis in joint:
            joint[axis] = joint[axis] & mask
        else:
            joint[axis] = mask
    slices = []
    for i in range(max(joint.keys()) + 1):
        pts = np.where(joint[i])[0][0]
        slices.append(pts)
    return tuple(slices)


def time_mask(times, time):
    """Logical mask that selects particular time"""
    if isinstance(time, (str, dt.datetime)):
        time = np.datetime64(time, 's')
    if isinstance(times, list):
        times = np.array(times, dtype='datetime64[s]')
    return times == time


def pressure_mask(pressures, pressure, rtol=0.01):
    """Logical mask that selects particular pressure"""
    if isinstance(pressures, list):
        pressures = np.array(pressures, dtype='d')
    return (np.abs(pressures - pressure) / np.abs(pressure)) < rtol


def pressure_axis(path, variable):
    return _axis("pressure", path, variable)


def time_axis(path, variable):
    return _axis("time", path, variable)


def _axis(name, path, variable):
    with netCDF4.Dataset(path) as dataset:
        var = dataset.variables[variable]
        for i, d in enumerate(var.dimensions):
            if d.startswith(name):
                return i
        coords = var.coordinates.split()
        for c in coords:
            if c.startswith(name):
                return 0
    msg = "{} axis not found: '{}' '{}'".format(name.capitalize(), path, variable)
    raise AxisNotFound(msg)


class Locator(object):
    """Locator for collection of UM diagnostic files

    Uses file naming convention and meta-data stored in
    files to quickly look up file/index related to point
    in space/time
    """
    def __init__(self, paths):
        self.paths = {}
        for path in paths:
            key = str(load_initial_time(path))
            if key in self.paths:
                self.paths[key].append(path)
            else:
                self.paths[key] = [path]

    def locate(
            self,
            pattern,
            variable,
            initial_time,
            valid_time,
            pressure=None,
            tolerance=0.001):
        if isinstance(valid_time, str):
            valid_time = np.datetime64(valid_time, 's')
        paths = self.paths[str(initial_time)]
        paths = fnmatch.filter(paths, pattern)
        for path in paths:
            valid_times = load_valid_times(path, variable)
            axes = [time_axis(path, variable)]
            masks = [time_mask(valid_times, valid_time)]
            if pressure is not None:
                try:
                    pressures = load_pressures(path, variable)
                    axis = pressure_axis(path, variable)
                    masks.append(pressure_mask(pressures, pressure))
                    axes.append(axis)
                except PressuresNotFound:
                    pass
            pts = ndindex(masks, axes)
            print(path, pts)
            return path, pts
        raise SearchFail('initial: {} valid: {} pressure: {}'.format(
            initial_time, valid_time, pressure))
