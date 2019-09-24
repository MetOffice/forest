"""Helpers to locate data on disk"""
import netCDF4
import datetime as dt
import numpy as np


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
