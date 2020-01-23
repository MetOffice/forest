import pytest
from forest.parse_args import parse_args


def test_no_files_or_config_raises_system_exit():
    with pytest.raises(SystemExit):
        parse_args([])


@pytest.mark.parametrize("argv,attr,expect", [
    (["file.json"], "files", ["file.json"]),
    (["--file-type", "rdt", "file.json"], "file_type", "rdt"),
    (["--config-file", "file.yml"], "config_file", "file.yml"),
    (["file.nc"], "variables", None),
    (["--var", "key", "value", "file.nc"], "variables", [["key", "value"]]),
    (["--var", "a", "b:c", "file.nc"], "variables", [["a", "b:c"]]),
    (["--var", "a", "b",
      "--var", "c", "d", "file.nc"], "variables", [["a", "b"], ["c", "d"]])
])
def test_parse_args(argv, attr, expect):
    result = getattr(parse_args(argv), attr)
    assert expect == result
