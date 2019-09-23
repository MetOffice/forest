import iris
import numpy as np
import netCDF4


class Coordinates(object):
    """Coordinate system for unified model diagnostics"""
    def initial_time(self, path):
        return InitialTimeLocator()(path)

    def variables(self, path):
        return []

    def valid_times(self, path, variable):
        return ValidTimesLocator()(path, variable)

    def pressures(self, path, variable):
        return PressuresLocator()(path, variable)


class InitialTimeLocator(object):
    def __call__(self, path):
        try:
            return self.netcdf4_strategy(path)
        except KeyError:
            return self.cube_strategy(path)

    @staticmethod
    def netcdf4_strategy(path):
        with netCDF4.Dataset(path) as dataset:
            var = dataset.variables["forecast_reference_time" ]
            values = netCDF4.num2date(var[:], units=var.units)
        return values

    @staticmethod
    def cube_strategy(path):
        cubes = iris.load(path)
        if len(cubes) > 0:
            cube = cubes[0]
            return cube.coord('time').cells().next().point
        raise InitialTimeNotFound("No initial time: '{}'".format(path))


class ValidTimesLocator(object):
    def __call__(self, path, variable):
        try:
            t = self.netcdf4_strategy(path, variable)
        except KeyError:
            t = self.cube_strategy(path, variable)
        if t is None:
            t = self.cube_strategy(path, variable)
        elif t.ndim == 0:
            t = np.array([t], dtype='datetime64[s]')
        return t

    def netcdf4_strategy(self, path, variable):
        with netCDF4.Dataset(path) as dataset:
            values = self._valid_times(dataset, variable)
        return values

    @staticmethod
    def _valid_times(dataset, variable):
        """Search dataset for time axis"""
        var = dataset.variables[variable]
        for d in var.dimensions:
            if d.startswith('time'):
                if d in dataset.variables:
                    tvar = dataset.variables[d]
                    return np.array(
                        netCDF4.num2date(tvar[:], units=tvar.units),
                        dtype='datetime64[s]')
        coords = var.coordinates.split()
        for c in coords:
            if c.startswith('time'):
                tvar = dataset.variables[c]
                return np.array(
                    netCDF4.num2date(tvar[:], units=tvar.units),
                    dtype='datetime64[s]')

    @staticmethod
    def cube_strategy(path, variable):
        cube = iris.load_cube(path, variable)
        return np.array([
            c.point for c in cube.coord('time').cells()],
                 dtype='datetime64[s]')


class PressuresLocator(object):
    def __call__(self, path, variable):
        try:
            return self.netcdf4_strategy(path, variable)
        except KeyError:
            return self.cube_strategy(path, variable)

    def cube_strategy(self, path, variable):
        try:
            cube = iris.load_cube(path, variable)
            points = cube.coord('pressure').points
            if np.ndim(points) == 0:
                points = np.array([points])
            return points
        except iris.exceptions.CoordinateNotFoundError:
            raise PressuresNotFound("'{}' '{}'".format(path, variable))

    @staticmethod
    def netcdf4_strategy(path, variable):
        """Search dataset for pressure axis"""
        with netCDF4.Dataset(path) as dataset:
            var = dataset.variables[variable]
            for d in var.dimensions:
                if d.startswith('pressure'):
                    if d in dataset.variables:
                        return dataset.variables[d][:]
            coords = var.coordinates.split()
            for c in coords:
                if c.startswith('pressure'):
                    return dataset.variables[c][:]
        # NOTE: refactor needed
        raise KeyError
