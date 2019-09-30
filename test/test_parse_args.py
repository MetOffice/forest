import unittest
from forest.parse_args import parse_args


class TestParseArgs(unittest.TestCase):
    def test_directory_returns_none_by_default(self):
        argv = [
            "--database", "file.db",
            "--config-file", "file.yml"]
        self.check(argv, "directory", None)

    def test_directory_returns_value(self):
        argv = [
            "--directory", "/some",
            "--database", "file.db",
            "--config-file", "file.yml"]
        self.check(argv, "directory", "/some")

    def test_files(self):
        self.check(["file.json"], "files", ["file.json"])

    def test_file_type(self):
        self.check(["--file-type", "rdt", "file.json"], "file_type", "rdt")

    def test_no_files_or_config_raises_system_exit(self):
        with self.assertRaises(SystemExit):
            parse_args([])

    def check(self, argv, attr, expect):
        result = getattr(parse_args(argv), attr)
        self.assertEqual(expect, result)
