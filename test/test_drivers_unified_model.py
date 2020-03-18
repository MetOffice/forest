import bokeh.models
import forest.drivers
import forest.db
import sqlite3


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
