import pytest
import unittest
import forest.cli.main
from forest.cli.main import parse_args


class TestForestCLI(unittest.TestCase):
    def test_parse_args_extra(self):
        _, extra = forest.cli.main.parse_args(["file.nc"])
        result = extra
        expect = ["file.nc"]
        self.assertEqual(expect, result)

    def test_parse_args_given_no_files_requires_config_file(self):
        with self.assertRaises(SystemExit):
            forest.cli.main.parse_args([])


@pytest.mark.parametrize("argv,expect", [
    (["--allow-websocket-origin", "eld388:8080", "file.nc"],
     ["bokeh", "serve", "/app/path",
                "--allow-websocket-origin", "eld388:8080",
                "--args", "file.nc"]),
    (["--dev", "file.nc"],
     ["bokeh", "serve", "/app/path",
                "--dev",
                "--args", "file.nc"]),
    (["--port", "5006", "file.nc"],
     ["bokeh", "serve", "/app/path",
                "--port", "5006",
                "--args", "file.nc"]),
    (["--show", "file.nc"],
     ["bokeh", "serve", "/app/path",
                "--show",
                "--args", "file.nc"]),
    (["--show", "--database", "file.db", "file.nc"],
     ["bokeh", "serve", "/app/path",
                "--show",
                "--args", "--database", "file.db", "file.nc"]),
    ])
def test_bokeh_args(argv, expect):
    args, extra = forest.cli.main.parse_args(argv)
    result = forest.cli.main.bokeh_args("/app/path", args, extra)
    assert expect == result
