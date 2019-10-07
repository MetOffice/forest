import os
import yaml
import sqlite3
import pytest
import forest
from forest import main


def test_main_given_rdt_files(tmp_path):
    rdt_file = tmp_path / "file.json"
    with rdt_file.open("w"):
        pass
    main.main(argv=["--file-type", "rdt", str(rdt_file)])


def test_file_groups_given_config_file(tmpdir):
    label = "UM"
    pattern = "*.nc"
    directory = "/dir"
    config_file = str(tmpdir / "config.yml")
    settings = {
        "files": [
            {
                "label": label,
                "pattern": pattern,
                "directory": directory,
                "locator": "database"
            }
        ]
    }
    with open(config_file, "w") as stream:
        yaml.dump(settings, stream)

    config = forest.config.load_config(config_file)
    actual = config.file_groups[0]
    expected = forest.config.FileGroup(
            label,
            pattern,
            directory=directory,
            locator="database")
    assert actual == expected
