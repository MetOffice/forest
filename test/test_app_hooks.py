import pytest
from unittest.mock import patch
import forest.parse_args
import forest.app_hooks


@pytest.mark.parametrize("argv,expect", [
    pytest.param(["bokeh", "serve",
                  "--use-xheaders",
                  "--port=1234",
                  "--allow-websocket-origin='*'",
                  "--args",
                  "--config-file", "file.yaml"],
                 ["--config-file", "file.yaml"], id="bokeh serve"),
    pytest.param(["forest",
                  "--port=1234",
                  "--allow-websocket-origin='*'",
                  "--config-file", "file.yaml"],
                 ["--config-file", "file.yaml"], id="forest"),
])
def test_parse_forest_args(argv, expect):
    """bokeh serve used when running in Docker container"""
    assert forest.app_hooks.parse_forest_args(argv) == expect


def test_parse_forest_args_given_none():
    with patch("forest.app_hooks.sys") as sys:
        sys.argv = ["bokeh", "serve", "--args", "--config-file", "file.yaml"]
        result = forest.app_hooks.parse_forest_args()
        assert result == ["--config-file", "file.yaml"]


def test_forest_main_parse_args():
    config_file = "file.yaml"
    args = forest.parse_args.parse_args(["forest",
                                         "--config-file", config_file])
    assert args.config_file == config_file
