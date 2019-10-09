import os
import shutil
import datetime as dt
import netCDF4
import numpy as np
import forest.db


SOURCE_DIR = os.path.dirname(__file__)
MULTI_CFG_FILE = "multi-config.yaml"
UM_CFG_FILE = "um-config.yaml"
UM_FILE = "unified_model.nc"
DB_FILE = "database.db"
RDT_FILE = "rdt_201904171245.json"
EIDA50_FILE = "eida50_20190417.nc"


def build_all(build_dir):
    """Build sample files"""
    for builder in [
            build_um_config,
            build_multi_config,
            build_rdt,
            build_eida50,
            build_um,
            build_database]:
        builder(build_dir)


def build_rdt(build_dir):
    build_file(build_dir, RDT_FILE)


def build_eida50(build_dir):
    build_file(build_dir, EIDA50_FILE)


def build_file(directory, file_name):
    src = os.path.join(SOURCE_DIR, file_name)
    dst = os.path.join(directory, file_name)
    print("copying: {} to {}".format(src, dst))
    shutil.copy2(src, dst)


def build_um_config(build_dir):
    path = os.path.join(build_dir, UM_CFG_FILE)
    content = """
files:
   - label: Unified Model
     pattern: "*{}"
     directory: {}
     locator: database
""".format(UM_FILE, build_dir)
    print("writing: {}".format(path))
    with open(path, "w") as stream:
        stream.write(content)


def build_multi_config(build_dir):
    path = os.path.join(build_dir, MULTI_CFG_FILE)
    content = """
files:
   - label: UM
     pattern: "unified_model*.nc"
     locator: file_system
     file_type: unified_model
   - label: EIDA50
     pattern: "eida50*.nc"
     locator: file_system
     file_type: eida50
   - label: RDT
     pattern: "rdt*.json"
     locator: file_system
     file_type: rdt
"""
    print("writing: {}".format(path))
    with open(path, "w") as stream:
        stream.write(content)


def build_um(build_dir):
    nx, ny = 100, 100
    x = np.linspace(0, 45, nx)
    y = np.linspace(0, 45, ny)
    X, Y = np.meshgrid(x, y)
    Z_0 = np.sqrt(X**2 + Y**2)
    Z_1 = Z_0 + 5.
    reference = dt.datetime(2019, 4, 17)
    times = [dt.datetime(2019, 4, 17, 12, 45), dt.datetime(2019, 4, 17, 13, 45)]
    path = os.path.join(build_dir, UM_FILE)
    print("writing: {}".format(path))
    with netCDF4.Dataset(path, "w") as dataset:
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
        var[0] = Z_0.T
        var[1] = Z_1.T


def build_database(build_dir):
    db_path = os.path.join(build_dir, DB_FILE)
    um_path = os.path.join(build_dir, UM_FILE)
    if not os.path.exists(um_path):
        build_um(build_dir)
    print("building: {}".format(db_path))
    database = forest.db.Database.connect(db_path)
    database.insert_netcdf(um_path)
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
