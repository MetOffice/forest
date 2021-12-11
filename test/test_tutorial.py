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
def test_main_calls_build_all_with_build_dir(tmpdir, file_names):
    if isinstance(file_names, str):
        file_names = [file_names]
    build_dir = str(tmpdir)

    runner.invoke(app, ["tutorial", build_dir])

    for file_name in file_names:
        assert os.path.exists(os.path.join(build_dir, file_name)), file_name


def test_rdt_file_name():
    assert forest.tutorial.core.RDT_FILE == "rdt_201904171245.json"


def test_build_rdt_copies_rdt_file_to_directory(tmpdir):
    directory = str(tmpdir)
    forest.tutorial.core.build_rdt(directory)
    expect = os.path.join(
        directory, os.path.basename(forest.tutorial.core.RDT_FILE)
    )
    assert os.path.exists(expect)


def test_build_all_makes_global_um_output(tmpdir):
    build_dir = str(tmpdir)
    forest.tutorial.core.build_all(build_dir)
    path = os.path.join(build_dir, forest.tutorial.core.UM_FILE)
    with netCDF4.Dataset(path) as dataset:
        var = dataset.variables["relative_humidity"]


def test_build_all_makes_eida50_example(tmpdir):
    build_dir = str(tmpdir)
    forest.tutorial.core.build_all(build_dir)
    path = os.path.join(build_dir, forest.tutorial.core.EIDA50_FILE)
    with netCDF4.Dataset(path) as dataset:
        var = dataset.variables["data"]


def test_build_all_makes_sample_database(tmpdir):
    build_dir = str(tmpdir)
    forest.tutorial.core.build_all(build_dir)
    file_name = os.path.join(build_dir, forest.tutorial.core.DB_FILE)
    assert os.path.exists(file_name)


def test_build_database_adds_sample_nc_to_file_table(tmpdir):
    build_dir = str(tmpdir)
    forest.tutorial.core.build_database(build_dir)
    db_path = os.path.join(build_dir, forest.tutorial.core.DB_FILE)
    um_path = os.path.join(build_dir, forest.tutorial.core.UM_FILE)
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM file")
    result = cursor.fetchall()
    expect = [(um_path,)]
    assert expect == result


def test_build_all_builds_um_config_file(tmpdir):
    build_dir = str(tmpdir)
    forest.tutorial.core.build_all(build_dir)
    path = str(tmpdir / forest.tutorial.core.UM_CFG_FILE)
    with open(path) as stream:
        result = yaml.safe_load(stream)
    expect = {
        "files": [
            {
                "label": "Unified Model",
                "pattern": "*" + forest.tutorial.core.UM_FILE,
                "directory": build_dir,
                "locator": "database",
            }
        ]
    }
    assert expect == result


def test_build_all_builds_config_file(tmpdir):
    build_dir = str(tmpdir)
    forest.tutorial.core.build_all(build_dir)
    path = str(tmpdir / forest.tutorial.core.MULTI_CFG_FILE)
    with open(path) as stream:
        result = yaml.safe_load(stream)
    expect = {
        "files": [
            {
                "label": "UM",
                "pattern": "unified_model*.nc",
                "locator": "file_system",
                "file_type": "unified_model",
            },
            {
                "label": "EIDA50",
                "pattern": "eida50*.nc",
                "locator": "file_system",
                "file_type": "eida50",
            },
            {
                "label": "RDT",
                "pattern": "rdt*.json",
                "locator": "file_system",
                "file_type": "rdt",
            },
        ]
    }
    assert expect == result


def test_build_name_config_file(tmpdir):
    build_dir = str(tmpdir)
    forest.tutorial.core.build_all(build_dir)
    builder = forest.tutorial.core.BUILDERS["name"]
    path = str(tmpdir / builder.file_name)
    with open(path) as stream:
        result = yaml.safe_load(stream)
    expect = {
        "files": [
            {"label": "NAME", "pattern": "NAME/*.txt", "file_type": "name"}
        ]
    }
    assert expect == result
