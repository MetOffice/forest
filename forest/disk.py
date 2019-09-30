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
        if axis in joint:
            joint[axis] = joint[axis] & mask
        else:
            joint[axis] = mask
    rank = max(joint.keys()) + 1  # find highest dimension
    return axes_pts([joint[i] for i in range(rank)])


def axes_pts(masks):
    slices = []
    for mask in masks:
        pts = np.where(mask)[0][0]
        slices.append(pts)
    return tuple(slices)


def coord_mask(name, values, value):
    return {
        "time": time_mask,
        "pressure": pressure_mask}[name](values, value)


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
    dims, coords = load_dim_coords(path, variable)
    value = axis(name, dims, coords)
    if value is None:
        msg = "{} axis not found: '{}' '{}'".format(name.capitalize(), path, variable)
        raise AxisNotFound(msg)
    else:
        return value


def load_dim_coords(path, variable):
    with netCDF4.Dataset(path) as dataset:
        var = dataset.variables[variable]
        dims = var.dimensions
        coords = getattr(var, "coordinates", "")
    return dims, coords


def has_coord(coord, dims, coords):
    return coord_var(coord, dims, coords) is not None


def coord_var(coord, dims, coords):
    for d in dims:
        if d.startswith(coord):
            return d
    for c in coords.split():
        if c.startswith(coord):
            return c


def axis(name, dims, coords):
    for i, d in enumerate(dims):
        if d.startswith(name):
            return i
    for c in coords.split():
        if c.startswith(name):
            return 0
