import pytest
from forest.parse_args import parse_args


def test_no_files_or_config_raises_system_exit():
    with pytest.raises(SystemExit):
        parse_args([])


def test_directory_returns_none_by_default():
    argv = [
        "--database", "file.db",
        "--config-file", "file.yml"]
    check(argv, "directory", None)


def test_directory_returns_value():
    argv = [
        "--directory", "/some",
        "--database", "file.db",
        "--config-file", "file.yml"]
    check(argv, "directory", "/some")


def test_files():
    check(["file.json"], "files", ["file.json"])


def test_file_type():
    check(["--file-type", "rdt", "file.json"], "file_type", "rdt")


def check(argv, attr, expect):
    result = getattr(parse_args(argv), attr)
    assert expect == result
