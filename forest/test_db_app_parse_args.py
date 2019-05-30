import unittest
import app.main


class TestApp(unittest.TestCase):
    def test_parse_args_requires_database(self):
        args = app.main.parse_args([
            "--database", "file.db",
            "--config-file", "file.yaml"
        ])
        self.assertEqual(args.database, "file.db")

    def test_parse_args_requires_config_file(self):
        args = app.main.parse_args([
            "--database", "file.db",
            "--config-file", "file.yaml"
        ])
        self.assertEqual(args.config_file, "file.yaml")
