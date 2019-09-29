import os
import datetime as dt
import netCDF4
import yaml
import numpy as np
import forest.db


SAMPLE_CFG = os.path.join(os.path.dirname(__file__), "sample.yaml")
SAMPLE_NC = os.path.join(os.path.dirname(__file__), "sample.nc")
SAMPLE_DB = os.path.join(os.path.dirname(__file__), "sample.db")
FILES = {
        "SAMPLE_CFG": SAMPLE_CFG,
        "SAMPLE_DB": SAMPLE_DB,
        "SAMPLE_NC": SAMPLE_NC
}


def build_all():
    """Build sample files"""
    for builder in [
            build_config,
            build_netcdf,
            build_database]:
        builder()


def build_config():
    data = {
        "files": [
            {
                "label": "SAMPLE",
                "pattern": SAMPLE_NC,
                "locator": "database"
            }
        ]
    }
    with open(SAMPLE_CFG, "w") as stream:
        yaml.dump(data, stream)


def build_netcdf():
    nx, ny = 100, 100
    x = np.linspace(0, 45, nx)
    y = np.linspace(0, 45, ny)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt(X**2 + Y**2)
    times = [dt.datetime(2019, 1, 1), dt.datetime(2019, 1, 2)]
    with netCDF4.Dataset(SAMPLE_NC, "w") as dataset:
        formatter = UM(dataset)
        var = formatter.longitudes(nx)
        var[:] = x
        var = formatter.latitudes(ny)
        var[:] = y
        var = formatter.times("time", length=len(times), dim_name="dim0")
        var[:] = netCDF4.date2num(times, units=var.units)
        formatter.forecast_reference_time(times[0])
        var = formatter.pressures("pressure", length=len(times), dim_name="dim0")
        var[:] = 1000.
        dims = ("dim0", "longitude", "latitude")
        coordinates = "forecast_period_1 forecast_reference_time pressure time"
        var = formatter.relative_humidity(dims, coordinates=coordinates)
        var[:] = Z.T


def build_database():
    database = forest.db.Database.connect(SAMPLE_DB)
    database.insert_netcdf(SAMPLE_NC)
    database.close()


class UM(object):
    """Unified model diagnostics formatter"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.units = "hours since 1970-01-01 00:00:00"

    def times(self, name, length=None, dim_name=None):
        if dim_name is None:
            dim_name = name
        dataset = self.dataset
        if dim_name not in dataset.dimensions:
            dataset.createDimension(dim_name, length)
        var = dataset.createVariable(name, "d", (dim_name,))
        var.axis = "T"
        var.units = self.units
        var.standard_name = "time"
        var.calendar = "gregorian"
        return var

    def forecast_reference_time(self, time, name="forecast_reference_time"):
        dataset = self.dataset
        var = dataset.createVariable(name, "d", ())
        var.units = self.units
        var.standard_name = name
        var.calendar = "gregorian"
        var[:] = netCDF4.date2num(time, units=self.units)

    def pressures(self, name, length=None, dim_name=None):
        if dim_name is None:
            dim_name = name
        dataset = self.dataset
        if dim_name not in dataset.dimensions:
            dataset.createDimension(dim_name, length)
        var = dataset.createVariable(name, "d", (dim_name,))
        var.axis = "Z"
        var.units = "hPa"
        var.long_name = "pressure"
        return var

    def longitudes(self, length=None, name="longitude"):
        dataset = self.dataset
        if name not in dataset.dimensions:
            dataset.createDimension(name, length)
        var = dataset.createVariable(name, "f", (name,))
        var.axis = "X"
        var.units = "degrees_east"
        var.long_name = "longitude"
        return var

    def latitudes(self, length=None, name="latitude"):
        dataset = self.dataset
        if name not in dataset.dimensions:
            dataset.createDimension(name, length)
        var = dataset.createVariable(name, "f", (name,))
        var.axis = "Y"
        var.units = "degrees_north"
        var.long_name = "latitude"
        return var

    def relative_humidity(self, dims, name="relative_humidity",
            coordinates="forecast_period_1 forecast_reference_time"):
        dataset = self.dataset
        var = dataset.createVariable(name, "f", dims)
        var.standard_name = "relative_humidity"
        var.units = "%"
        var.um_stash_source = "m01s16i204"
        var.grid_mapping = "latitude_longitude"
        var.coordinates = coordinates
        return var
