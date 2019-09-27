import unittest
import forest.app.main as main


class TestApp(unittest.TestCase):
    def test_parse_args_requires_database(self):
        args = main.parse_args([
            "--database", "file.db",
            "--config-file", "file.yaml"
        ])
        self.assertEqual(args.database, "file.db")

    def test_parse_args_requires_config_file(self):
        args = main.parse_args([
            "--database", "file.db",
            "--config-file", "file.yaml"
        ])
        self.assertEqual(args.config_file, "file.yaml")
