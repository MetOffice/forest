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


def test_replace_dir_given_args_dir_only():
    actual = main.replace_dir("args/dir", None)
    expected = "args/dir"
    assert actual == expected


def test_replace_dir_given_group_dir_only():
    actual = main.replace_dir(None, "group/dir")
    expected = "group/dir"
    assert actual == expected


def test_replace_dir_given_relative_group_dir_appends_to_args_dir():
    actual = main.replace_dir("args/dir", "leaf")
    expected = "args/dir/leaf"
    assert actual == expected


def test_replace_dir_given_absolute_group_dir_overrides_rel_args_dir():
    actual = main.replace_dir("args/relative", "/group/absolute")
    expected = "/group/absolute"
    assert actual == expected


def test_replace_dir_given_absolute_group_dir_overrides_abs_args_dir():
    actual = main.replace_dir("/args/absolute", "/group/absolute")
    expected = "/group/absolute"
    assert actual == expected
