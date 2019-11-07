import pytest
from forest.parse_args import parse_args


def test_no_files_or_config_raises_system_exit():
    with pytest.raises(SystemExit):
        parse_args([])


@pytest.mark.parametrize("argv,attr,expect", [
    (["file.json"], "files", ["file.json"]),
    (["--file-type", "rdt", "file.json"], "file_type", "rdt"),
    (["--directory", "/some",
      "--database", "file.db",
      "--config-file", "file.yml"], "directory", "/some"),
    (["--database", "file.db",
      "--config-file", "file.yml"], "directory", None)
])
def test_parse_args(argv, attr, expect):
    result = getattr(parse_args(argv), attr)
    assert expect == result
