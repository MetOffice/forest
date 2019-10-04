import os
import sqlite3
import pytest
import forest
from forest import main


def test_main_given_rdt_files(tmp_path):
    rdt_file = tmp_path / "file.json"
    with rdt_file.open("w"):
        pass
    main.main(argv=["--file-type", "rdt", str(rdt_file)])


def test_build_loader():
    group = forest.config.FileGroup("name", "pattern")
    args = main.parse_args.parse_args(["file_20190101T0000Z.nc"])
    main.build_loader(group, args)
