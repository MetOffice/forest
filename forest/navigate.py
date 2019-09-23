import numpy as np
import fnmatch
from .exceptions import (
        InitialTimeNotFound,
        ValidTimesNotFound,
        PressuresNotFound)
from . import (
        unified_model,
        eida50,
        rdt)


class FileSystem(object):
    """Navigates collections of file(s)

    .. note:: This is a naive implementation designed
              to support basic command line file usage
    """
    def __init__(self, paths, coordinates=None):
        self.paths = paths
        if coordinates is None:
            coordinates = unified_model.Coordinates()
        self.coordinates = coordinates

    @classmethod
    def file_type(cls, paths, file_type):
        if file_type.lower() == "rdt":
            coordinates = rdt.Coordinates()
        elif file_type.lower() == "eida50":
            coordinates = eida50.Coordinates()
        elif file_type.lower() == "unified_model":
            coordinates = unified_model.Coordinates()
        else:
            raise Exception("Unrecognised file type: '{}'".format(file_type))
        return cls(paths, coordinates)

    def variables(self, pattern):
        paths = fnmatch.filter(self.paths, pattern)
        names = []
        for path in paths:
            names += self.coordinates.variables(path)
        return list(sorted(set(names)))

    def initial_times(self, pattern, variable=None):
        paths = fnmatch.filter(self.paths, pattern)
        times = []
        for path in paths:
            try:
                time = self.coordinates.initial_time(path)
                if time is None:
                    continue
                times.append(time)
            except InitialTimeNotFound:
                pass
        return list(sorted(set(times)))

    def valid_times(self, pattern, variable, initial_time):
        paths = fnmatch.filter(self.paths, pattern)
        arrays = []
        for path in paths:
            try:
                array = self.coordinates.valid_times(path, variable)
                if array is None:
                    continue
                arrays.append(array)
            except ValidTimesNotFound:
                pass
        if len(arrays) == 0:
            return []
        return np.unique(np.concatenate(arrays))

    def pressures(self, pattern, variable, initial_time):
        paths = fnmatch.filter(self.paths, pattern)
        arrays = []
        for path in paths:
            try:
                array = self.coordinates.pressures(path, variable)
                if array is None:
                    continue
                arrays.append(array)
            except PressuresNotFound:
                pass
        if len(arrays) == 0:
            return []
        return np.unique(np.concatenate(arrays))
