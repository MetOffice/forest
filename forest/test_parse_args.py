import unittest
import parse_args


class TestParseArgs(unittest.TestCase):
    def test_directory_returns_none_by_default(self):
        args = parse_args.parse_args([
            "--database", "file.db",
            "--config-file", "file.yml"
        ])
        result = args.directory
        expect = None
        self.assertEqual(expect, result)

    def test_directory_returns_value(self):
        args = parse_args.parse_args([
            "--directory", "/some",
            "--database", "file.db",
            "--config-file", "file.yml"
        ])
        result = args.directory
        expect = "/some"
        self.assertEqual(expect, result)
