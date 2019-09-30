import unittest
import forest.cli.main
from forest.cli.main import parse_args


class TestForestCLI(unittest.TestCase):
    def test_files(self):
        paths = ["file.json"]
        result = getattr(parse_args(paths), "files")
        expect = paths
        self.assertEqual(expect, result)

    def test_parse_args(self):
        namespace = forest.cli.main.parse_args(["file.nc"])
        result = namespace.files
        expect = ["file.nc"]
        self.assertEqual(expect, result)

    def test_parse_args_given_no_files_requires_config_file(self):
        with self.assertRaises(SystemExit):
            forest.cli.main.parse_args([])


class TestForestToBokehServe(unittest.TestCase):
    def setUp(self):
        self.app_path = "/app/path"

    def test_bokeh_args_given_show(self):
        self.check_serve_args(["--show"])

    def test_bokeh_args_given_port(self):
        self.check_serve_args(["--port", "5006"])

    def test_bokeh_args_given_dev(self):
        self.check_serve_args(["--dev"])

    def test_bokeh_args_given_allow_websocket_origin(self):
        self.check_serve_args(["--allow-websocket-origin", "eld388:8080"])

    def check_serve_args(self, opts):
        args = forest.cli.main.parse_args(opts + ["file.nc"])
        result = forest.cli.main.bokeh_args(self.app_path, args)
        expect = [
            "bokeh",
            "serve",
            "/app/path"] + opts + [
            "--args",
            "file.nc"
        ]
        self.assertEqual(expect, result)

    def test_bokeh_args_given_config(self):
        self.check_extra_args(["--config", "file.yml"],
                              ["--config-file", "file.yml"])

    def test_bokeh_args_given_database(self):
        self.check_extra_args(["--database", "file.db", "file.nc"])

    def test_bokeh_args_given_directory(self):
        self.check_extra_args(["--directory", "/some/prefix", "file.nc"])

    def check_extra_args(self, given_opts, expect_opts=None):
        if expect_opts is None:
            expect_opts = given_opts
        args = forest.cli.main.parse_args(given_opts)
        result = forest.cli.main.bokeh_args(self.app_path, args)
        expect = [
            "bokeh",
            "serve",
            "/app/path",
            "--args"] + expect_opts
        self.assertEqual(expect, result)
