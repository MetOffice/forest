"""
Sample data should be available to let users/developers build
extensions easily
"""
import os
import netCDF4
import sqlite3
import yaml
import forest


def test_parse_args_build_dir():
    args = forest.tutorial.main.parse_args(["build"])
    assert args.build_dir == "build"


def test_main_calls_build_all_with_build_dir(tmpdir):
    build_dir = str(tmpdir)
    forest.tutorial.main.main([build_dir])
    assert os.path.exists(os.path.join(build_dir, forest.tutorial.RDT_FILE))


def test_rdt_file_name():
    assert forest.tutorial.RDT_FILE == "rdt_201904171245.json"


def test_build_rdt_copies_rdt_file_to_directory(tmpdir):
    directory = str(tmpdir)
    forest.tutorial.build_rdt(directory)
    expect = os.path.join(directory, os.path.basename(forest.tutorial.RDT_FILE))
    assert os.path.exists(expect)


def test_build_all_makes_global_um_output(tmpdir):
    build_dir = str(tmpdir)
    forest.tutorial.build_all(build_dir)
    path = os.path.join(build_dir, forest.tutorial.UM_FILE)
    with netCDF4.Dataset(path) as dataset:
        var = dataset.variables["relative_humidity"]


def test_build_all_makes_eida50_example(tmpdir):
    build_dir = str(tmpdir)
    forest.tutorial.build_all(build_dir)
    path = os.path.join(build_dir, forest.tutorial.EIDA50_FILE)
    with netCDF4.Dataset(path) as dataset:
        var = dataset.variables["data"]


def test_build_all_makes_sample_database(tmpdir):
    build_dir = str(tmpdir)
    forest.tutorial.build_all(build_dir)
    file_name = os.path.join(build_dir, forest.tutorial.DB_FILE)
    assert os.path.exists(file_name)


def test_build_database_adds_sample_nc_to_file_table(tmpdir):
    build_dir = str(tmpdir)
    forest.tutorial.build_database(build_dir)
    db_path = os.path.join(build_dir, forest.tutorial.DB_FILE)
    um_path = os.path.join(build_dir, forest.tutorial.UM_FILE)
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM file")
    result = cursor.fetchall()
    expect = [(um_path,)]
    assert expect == result


def test_build_all_builds_um_config_file(tmpdir):
    build_dir = str(tmpdir)
    forest.tutorial.build_all(build_dir)
    path = str(tmpdir / forest.tutorial.UM_CFG_FILE)
    with open(path) as stream:
        result = yaml.safe_load(stream)
    expect = {
        "files": [{
            "label": "Unified Model",
            "pattern": "*" + forest.tutorial.UM_FILE,
            "directory": build_dir,
            "locator": "database"
        }]
    }
    assert expect == result


def test_build_all_builds_config_file(tmpdir):
    build_dir = str(tmpdir)
    forest.tutorial.build_all(build_dir)
    path = str(tmpdir / forest.tutorial.MULTI_CFG_FILE)
    with open(path) as stream:
        result = yaml.safe_load(stream)
    expect = {
        "files": [{
            "label": "UM",
            "pattern": "unified_model*.nc",
            "locator": "file_system",
            "file_type": "unified_model"
        }, {
            "label": "EIDA50",
            "pattern": "eida50*.nc",
            "locator": "file_system",
            "file_type": "eida50"
        }, {
            "label": "RDT",
            "pattern": "rdt*.json",
            "locator": "file_system",
            "file_type": "rdt"
        }]
    }
    assert expect == result
