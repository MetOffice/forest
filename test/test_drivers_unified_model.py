import forest.drivers
import forest.db
import sqlite3


def test_dataset():
    settings = {
        "pattern": "*.nc"
    }
    dataset = forest.drivers.get_dataset("unified_model", settings)


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
