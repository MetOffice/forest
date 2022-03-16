"""
Sample data should be available to let users/developers build
extensions easily
"""
import pytest
import os
import netCDF4
import sqlite3
import yaml
import forest.tutorial.core
from typer.testing import CliRunner
from forest.cli.alternative import app

runner = CliRunner()


@pytest.fixture
def build_dir(tmpdir):
    return str(tmpdir)


@pytest.mark.parametrize(
    "file_names",
    [
        pytest.param(
            forest.tutorial.core.RDT_FILE, id="Rapid developing thunderstorms"
        ),
        pytest.param(
            forest.tutorial.core.FILE_NAMES["NAME"], id="NAME dispersion data"
        ),
    ],
)
def test_main_calls_build_all_with_build_dir(build_dir, file_names):
    if isinstance(file_names, str):
        file_names = [file_names]

    runner.invoke(app, ["tutorial", build_dir])

    for file_name in file_names:
        assert os.path.exists(os.path.join(build_dir, file_name)), file_name


def test_rdt_file_name():
    assert forest.tutorial.core.RDT_FILE == "rdt_201904171245.json"


def test_build_rdt_copies_rdt_file_to_directory(build_dir):
    forest.tutorial.core.build_rdt(build_dir)
    expect = os.path.join(
        build_dir, os.path.basename(forest.tutorial.core.RDT_FILE)
    )
    assert os.path.exists(expect)


def test_build_all_makes_global_um_output(build_dir):
    forest.tutorial.core.build_all(build_dir)
    path = os.path.join(build_dir, forest.tutorial.core.UM_FILE)
    with netCDF4.Dataset(path) as dataset:
        var = dataset.variables["relative_humidity"]


def test_build_all_makes_eida50_example(build_dir):
    forest.tutorial.core.build_all(build_dir)
    path = os.path.join(build_dir, forest.tutorial.core.EIDA50_FILE)
    with netCDF4.Dataset(path) as dataset:
        var = dataset.variables["data"]


def test_build_all_makes_sample_database(build_dir):
    forest.tutorial.core.build_all(build_dir)
    file_name = os.path.join(build_dir, forest.tutorial.core.DB_FILE)
    assert os.path.exists(file_name)


def test_build_database_adds_sample_nc_to_file_table(build_dir):
    forest.tutorial.core.build_database(build_dir)
    db_path = os.path.join(build_dir, forest.tutorial.core.DB_FILE)
    um_path = os.path.join(build_dir, forest.tutorial.core.UM_FILE)
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM file")
    result = cursor.fetchall()
    expect = [(um_path,)]
    assert expect == result


def test_build_all_builds_um_config_file(build_dir):
    forest.tutorial.core.build_all(build_dir)
    path = os.path.join(build_dir, forest.tutorial.core.UM_CFG_FILE)
    with open(path) as stream:
        result = yaml.safe_load(stream)

    expect = {
        "edition": 2022,
        "datasets": [
            {
                "label": "Unified Model",
                "driver": {
                    "name": "unified_model",
                    "settings": {
                        "pattern": "*" + forest.tutorial.core.UM_FILE,
                        "directory": build_dir,
                        "locator": "database",
                        "database_path": "database.db",
                    },
                },
            }
        ],
    }
    assert expect == result


def test_build_all_builds_config_file(build_dir):
    forest.tutorial.core.build_all(build_dir)
    path = os.path.join(build_dir, forest.tutorial.core.MULTI_CFG_FILE)
    with open(path) as stream:
        result = yaml.safe_load(stream)
    expect = {
        "edition": 2022,
        "datasets": [
            {
                "label": "UM",
                "driver": {
                    "name": "unified_model",
                    "settings": {
                        "pattern": "unified_model*.nc",
                        "locator": "file_system",
                    },
                },
            },
            {
                "label": "EIDA50",
                "driver": {
                    "name": "eida50",
                    "settings": {
                        "pattern": "eida50*.nc",
                    },
                },
            },
            {
                "label": "RDT",
                "driver": {
                    "name": "rdt",
                    "settings": {
                        "pattern": "rdt*.json",
                    },
                },
            },
        ],
    }
    assert expect == result


def test_build_name_config_file(build_dir):
    # Build all assets
    forest.tutorial.core.build_all(build_dir)

    # Use NAME file builder to load config file
    builder = forest.tutorial.core.BUILDERS["name"]
    path = os.path.join(build_dir, builder.file_name)
    with open(path) as stream:
        result = yaml.safe_load(stream)

    # Modern configuration data structure
    expect = {
        "edition": 2022,
        "datasets": [
            {
                "label": "NAME",
                "driver": {
                    "name": "name",
                    "settings": {"pattern": "NAME/*.txt"},
                },
            }
        ],
    }
    assert expect == result
