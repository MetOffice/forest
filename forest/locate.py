"""Helper methods for locating file(s)"""
import numpy as np
import datetime as dt


def bounds(times, length):
    dtype = 'datetime64[s]'
    if isinstance(times, list):
        times = np.asarray(times, dtype=dtype)
    if isinstance(length, dt.timedelta):
        length = np.asarray(length, dtype='timedelta64[s]')
    return np.array([times, times + length], dtype=dtype).T


def in_bounds(bounds, point):
    if isinstance(point, str):
        point = np.datetime64(point, 's')
    return (bounds[:, 0] <= point) & (point < bounds[:, 1])
