import os
import json
import yaml
import sqlite3
import pytest
import forest
from forest import main


def test_main_given_rdt_files(tmp_path):
    rdt_file = tmp_path / "file_202001010000.json"
    with rdt_file.open("w") as stream:
        json.dump({
            "features": {}
        }, stream)
    main.main(argv=["--file-type", "rdt", str(rdt_file)])


def test_file_groups_given_config_file(tmpdir):
    label = "UM"
    pattern = "*.nc"
    config_file = str(tmpdir / "config.yml")
    settings = {
        "files": [
            {
                "label": label,
                "pattern": pattern,
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
            locator="database")
    assert actual == expected
