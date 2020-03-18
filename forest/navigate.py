import numpy as np
import fnmatch
import glob
import os
from .exceptions import (
        InitialTimeNotFound,
        ValidTimesNotFound,
        PressuresNotFound)
from forest import (
        exceptions,
        drivers,
        db,
        gridded_forecast,
        rdt,
        intake_loader,
        nearcast)


class Navigator:
    def __init__(self, config, color_mapper=None):
        # TODO: Once the idea of a "Group" exists we can avoid using the
        # config and defer the sub-navigator creation to each of the
        # groups. This will remove the need for the `_from_group` helper
        # and the logic in FileSystemNavigator.from_file_type().
        # Also, it'd be good to switch the identification of groups from
        # using the `pattern` to using the `label`. In general, not every
        # group would have a `pattern`.
        # e.g.
        # self._navigators = {group.label: group.navigator for group in ...}
        self._navigators = {group.pattern: self._from_group(group, color_mapper)
                           for group in config.file_groups}

    @classmethod
    def _from_group(cls, group, color_mapper=None):
        try:
            settings = {
                "label": group.label,
                "pattern": group.pattern,
                "locator": group.locator,
                "database_path": group.database_path,
                "color_mapper": color_mapper,
            }
            dataset = drivers.get_dataset(group.file_type, settings)
            return dataset.navigator()
        except exceptions.DriverNotFound:
            # TODO: Migrate all file types to forest.drivers
            pass

        paths = cls._expand_paths(group.pattern)
        navigator = FileSystemNavigator.from_file_type(paths,
                                                       group.file_type,
                                                       group.pattern)
        return navigator

    @classmethod
    def _expand_paths(cls, pattern):
        return glob.glob(os.path.expanduser(pattern))

    def variables(self, pattern):
        navigator = self._navigators[pattern]
        return navigator.variables(pattern)

    def initial_times(self, pattern, variable=None):
        navigator = self._navigators[pattern]
        return navigator.initial_times(pattern, variable=variable)

    def valid_times(self, pattern, variable, initial_time):
        navigator = self._navigators[pattern]
        return navigator.valid_times(pattern, variable, initial_time)

    def pressures(self, pattern, variable, initial_time):
        navigator = self._navigators[pattern]
        return navigator.pressures(pattern, variable, initial_time)


class FileSystemNavigator:
    """Navigates collections of file(s)

    .. note:: This is a naive implementation designed
              to support basic command line file usage
    """
    def __init__(self, paths, coordinates):
        self.paths = paths
        self.coordinates = coordinates

    @classmethod
    def from_file_type(cls, paths, file_type, pattern=None):
        if file_type.lower() == "rdt":
            coordinates = rdt.Coordinates()
            return cls(paths, coordinates)
        elif file_type.lower() == 'griddedforecast':
            # XXX This needs a "Group" object ... not "paths"
            return gridded_forecast.Navigator(paths)
        elif file_type.lower() == 'intake':
            return intake_loader.Navigator()
        elif file_type.lower() == "nearcast":
            return nearcast.Navigator(pattern)
        else:
            raise Exception("Unrecognised file type: '{}'".format(file_type))

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
