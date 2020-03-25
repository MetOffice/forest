import pytest
import bokeh.models
import forest.drivers
from forest.drivers import unified_model
import forest.db
import sqlite3
import netCDF4
import iris


def test_dataset_loader_pattern():
    settings = {
        "pattern": "*.nc",
        "color_mapper": bokeh.models.ColorMapper()
    }
    dataset = forest.drivers.get_dataset("unified_model", settings)
    view = dataset.map_view()
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
        "color_mapper": bokeh.models.ColorMapper()
    }
    dataset = forest.drivers.get_dataset("unified_model", settings)
    view = dataset.map_view()
    assert hasattr(view.loader.locator, "connection")
    assert view.loader.locator.directory == "/replace"


def test_load_image_pts(tmpdir):
    path = str(tmpdir / "file.nc")
    variable = "air_temperature"
    with netCDF4.Dataset(path, "w") as dataset:
        make_file(dataset, variable)
    data = unified_model.load_image_pts(path, variable, (), ())
    assert data["image"][0].shape == (2, 2)


def test_iris_load(tmpdir):
    path = str(tmpdir / "file.nc")
    variable = "air_temperature"
    with netCDF4.Dataset(path, "w") as dataset:
        make_file(dataset, variable)
    cubes = iris.load(path)
    names = [c.name() for c in cubes]
    assert names == [variable]


def make_file(dataset, variable):
    dimensions = [
        ("time", 1),
        ("longitude", 2),
        ("latitude", 2)
    ]
    for name, length in dimensions:
        dataset.createDimension(name, length)
    var = dataset.createVariable("longitude", "f", ("longitude",))
    var[:] = [0, 1]
    var = dataset.createVariable("latitude", "f", ("latitude",))
    var[:] = [0, 1]
    var = dataset.createVariable(variable, "f", ("longitude", "latitude"))
