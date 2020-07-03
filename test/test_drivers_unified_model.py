import pytest
import datetime as dt
import bokeh.models
import forest.drivers
from forest.drivers import unified_model
import forest.db
import sqlite3
import netCDF4
import iris


def test_sync_skips_oserror(tmpdir):
    """Files stored in S3 Glacier are inaccessible via goofys"""
    database_path = str(tmpdir / "file.db")
    connection = sqlite3.connect(database_path)
    connection.close()
    file_name = str(tmpdir / "file.nc")
    with open(file_name, "w"):
        # Simulate NetCDF files stored in S3 Glacier
        pass
    settings = {
        "locator": "database",
        "pattern": file_name,
        "directory": str(tmpdir),
        "database_path": database_path
    }
    dataset = forest.drivers.get_dataset("unified_model", settings)
    dataset.sync()


def test_dataset_loader_pattern():
    settings = {
        "pattern": "*.nc",
    }
    color_mapper = bokeh.models.ColorMapper()
    dataset = forest.drivers.get_dataset("unified_model", settings)
    view = dataset.map_view(color_mapper)
    assert isinstance(view.loader, forest.drivers.unified_model.Loader)


def test_navigator_use_database(tmpdir):
    database_path = str(tmpdir / "fake.db")
    connection = sqlite3.connect(database_path)
    connection.close()
    settings = {
        "pattern": "*.nc",
        "locator": "database",
        "database_path": database_path
    }
    dataset = forest.drivers.get_dataset("unified_model", settings)
    navigator = dataset.navigator()
    assert isinstance(navigator, forest.db.Database)


def test_loader_use_database(tmpdir):
    database_path = str(tmpdir / "database.db")
    connection = sqlite3.connect(database_path)
    connection.close()
    settings = {
        "label": "UM",
        "pattern": "*.nc",
        "directory": "/replace",
        "locator": "database",
        "database_path": database_path,
    }
    color_mapper = bokeh.models.ColorMapper()
    dataset = forest.drivers.get_dataset("unified_model", settings)
    view = dataset.map_view(color_mapper)
    assert hasattr(view.loader.locator, "connection")
    assert view.loader.locator.directory == "/replace"


def test_view_render_state(tmpdir):
    path = str(tmpdir / "file_20200101.nc")
    variable = "air_temperature"
    times = [dt.datetime(2020, 1, 1)]
    with netCDF4.Dataset(path, "w") as dataset:
        insert_lonlat(dataset, [0, 1], [0, 1])
        insert_times(dataset, times)
        var = dataset.createVariable(variable, "f", ("time", "longitude", "latitude"))
        var.coords = "time"  # Needed by Locator
    settings = {
        "pattern": path,
    }
    color_mapper = bokeh.models.ColorMapper()
    dataset = forest.drivers.get_dataset("unified_model", settings)
    view = dataset.map_view(color_mapper)
    view.render({
        "initial_time": dt.datetime(2020, 1, 1),
        "valid_time": dt.datetime(2020, 1, 1),
        "variable": variable,
        "pressure": None,
        "pressures": []  # Needed by Loader.valid(state)
    })
    assert len(view.image_sources[0].data["image"]) == 1


def test_load_image(tmpdir):
    path = str(tmpdir / "file.nc")
    variable = "air_temperature"
    with netCDF4.Dataset(path, "w") as dataset:
        insert_lonlat(dataset, [0, 1], [0, 1])
        var = dataset.createVariable(variable, "f", ("longitude", "latitude"))
    data = unified_model.Loader.load_image(path, variable, ())
    assert data["image"][0].shape == (2, 2)


def test_iris_load(tmpdir):
    path = str(tmpdir / "file.nc")
    variable = "air_temperature"
    with netCDF4.Dataset(path, "w") as dataset:
        insert_lonlat(dataset, [0, 1], [0, 1])
        var = dataset.createVariable(variable, "f", ("longitude", "latitude"))
    cubes = iris.load(path)
    names = [c.name() for c in cubes]
    assert names == [variable]


def insert_times(dataset, times):
    if "time" not in dataset.dimensions:
        dataset.createDimension("time", len(times))
    units = "seconds since 1970-01-01 00:00:00 utc"
    var = dataset.createVariable("time", "f", ("time",))
    var.units = units
    var[:] = netCDF4.date2num(times, units=units)


def insert_lonlat(dataset, lons, lats):
    dimensions = [
        ("longitude", len(lons)),
        ("latitude", len(lats))
    ]
    for name, length in dimensions:
        dataset.createDimension(name, length)
    var = dataset.createVariable("longitude", "f", ("longitude",))
    var[:] = lons
    var = dataset.createVariable("latitude", "f", ("latitude",))
    var[:] = lats
