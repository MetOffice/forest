"""
Sample data should be available to let users/developers build
extensions easily
"""
import os
import netCDF4
import sqlite3
import yaml
import forest


def test_sample_files():
    result = forest.sample.FILES
    expect = {
        "SAMPLE_NC": forest.sample.SAMPLE_NC,
        "SAMPLE_DB": forest.sample.SAMPLE_DB,
        "SAMPLE_CFG": forest.sample.SAMPLE_CFG
    }
    assert expect == result


def test_build_all_makes_global_um_output():
    forest.sample.build_all()
    with netCDF4.Dataset(forest.sample.SAMPLE_NC) as dataset:
        var = dataset.variables["relative_humidity"]


def test_build_all_makes_sample_database():
    file_name = forest.sample.SAMPLE_DB
    if os.path.exists(file_name):
        os.remove(file_name)
    forest.sample.build_all()
    assert os.path.exists(file_name)


def test_build_database_adds_sample_nc_to_file_table():
    forest.sample.build_database()
    connection = sqlite3.connect(forest.sample.SAMPLE_DB)
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM file")
    result = cursor.fetchall()
    expect = [(forest.sample.SAMPLE_NC,)]
    assert expect == result


def test_build_all_builds_config_file():
    forest.sample.build_all()
    with open(forest.sample.SAMPLE_CFG) as stream:
        result = yaml.load(stream)
    expect = {
        "files": [{
            "label": "SAMPLE",
            "pattern": forest.sample.SAMPLE_NC,
            "locator": "database"
        }]
    }
    assert expect == result
