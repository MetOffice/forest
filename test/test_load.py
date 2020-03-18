import pytest
import yaml
import bokeh.models
import forest
import forest.drivers
from forest import main, rdt


def test_rdt_loader_given_pattern():
    loader = forest.Loader.from_pattern("Label", "RDT*.json", "rdt")
    assert isinstance(loader, rdt.Loader)


def test_build_loader_given_files():
    settings = {"pattern": "file_20190101T0000Z.nc",
                "color_mapper": bokeh.models.ColorMapper()}
    dataset = forest.drivers.get_dataset("unified_model", settings)
    view = dataset.map_view()
    assert isinstance(view.loader, forest.data.DBLoader)
    assert isinstance(view.loader.locator, forest.drivers.unified_model.Locator)


@pytest.mark.skip()
def test_build_loader_given_database(tmpdir):
    """replicate main.py as close as possible"""
    database_file = str(tmpdir / "database.db")

    config_file = str(tmpdir / "config.yml")
    settings = {
        "files": [
            {
                "label": "UM",
                "pattern": "*.nc",
                "directory": "/replace",
                "locator": "database",
                "database_path": database_file
            }
        ]
    }
    with open(config_file, "w") as stream:
        yaml.dump(settings, stream)

    args = main.parse_args.parse_args([
        "--config-file", config_file])
    config = forest.config.load_config(args.config_file)
    group = config.file_groups[0]
    database = forest.db.Database.connect(database_file)
    loader = forest.Loader.group_args(group, args, database=database)
    database.close()
    assert hasattr(loader.locator, "connection")
    assert loader.locator.directory == "/replace"


@pytest.mark.skip()
def test_build_loader_given_config_file_pattern(tmpdir):
    config_file = str(tmpdir / "config.yml")
    path = str(tmpdir / "file_20190101T0000Z.nc")
    with open(path, "w"):
        pass
    args = main.parse_args.parse_args([
        "--config-file", config_file])
    label = "UM"
    pattern = str(tmpdir/ "file_*.nc")
    group = forest.config.FileGroup(
            label,
            pattern,
            locator="file_system")
    loader = forest.Loader.group_args(group, args)
    assert loader.locator.paths == [path]


def test_build_loader_given_rdt_file_type():
    loader = forest.Loader.from_pattern(
            "Label", "*.json", "rdt")
    assert isinstance(loader, forest.rdt.Loader)
    assert isinstance(loader.locator, forest.rdt.Locator)
