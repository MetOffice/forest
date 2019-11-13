import pytest
import unittest
import yaml
import os
import forest


@pytest.mark.parametrize("env,args,expected", [
        ({}, {}, {}),
        ({}, [], {}),
        ({}, {"k": "v"}, {"k": "v"}),
        ({"k": "v"}, {}, {"k": "v"}),
        ({"x": "environment"}, {"x": "user"}, {"x": "user"}),
        ({"x": "environment"}, [("x", "a"), ("y", "b")], {"x": "a", "y": "b"}),
        ({"z": "c"}, [("x", "a"), ("y", "b")], {"x": "a", "y": "b", "z": "c"}),
    ])
def test_config_combine_os_environ_with_args(env, args, expected):
    actual = forest.config.combine_variables(env, args)
    assert actual == expected


def test_combine_variables_copies_environment():
    forest.config.combine_variables(os.environ, dict(custom="value"))
    with pytest.raises(KeyError):
        os.environ["custom"]


def test_config_template_substitution(tmpdir):
    config_file = str(tmpdir / "test-config.yml")
    with open(config_file, "w") as stream:
        stream.write(yaml.dump({
            "parameter": "${X}/file.nc"
        }))
    variables = {
            "X": "/expand"
    }
    config = forest.config.Config.load(config_file, variables)
    assert config.data == {"parameter": "/expand/file.nc"}


class TestIntegration(unittest.TestCase):
    def setUp(self):
        path = os.path.join(os.path.dirname(__file__), '../forest/config.yaml')
        self.config = forest.load_config(path)

    def test_load_server_config_first_group(self):
        result = self.config.file_groups[0]
        expect = forest.config.FileGroup(
                "Operational GA6 Africa",
                "*global_africa*.nc",
                locator="database",
                directory="unified_model")
        self.assert_group_equal(expect, result)

    def test_load_server_config_second_group(self):
        result = self.config.file_groups[2]
        expect = forest.config.FileGroup(
                "Operational Tropical Africa",
                "*os42_ea*.nc",
                locator="database",
                directory="unified_model")
        self.assert_group_equal(expect, result)

    def test_load_server_config_has_eida50(self):
        groups = [g for g in self.config.file_groups
            if g.file_type == 'eida50']
        result = groups[0]
        expect = forest.config.FileGroup(
                "EIDA50",
                "eida50/EIDA50_takm4p4*.nc",
                file_type="eida50")
        self.assert_group_equal(expect, result)

    def assert_group_equal(self, expect, result):
        self.assertEqual(expect.label, result.label)
        self.assertEqual(expect.pattern, result.pattern)
        self.assertEqual(expect.locator, result.locator)
        self.assertEqual(expect, result)


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.path = "test-config.yaml"

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_load_config(self):
        data = {
            "files": [
                {"label": "EIDA50",
                 "directory": "~/cache",
                 "pattern": "*.nc"}
            ]
        }
        with open(self.path, "w") as stream:
            yaml.dump(data, stream)
        result = forest.load_config(self.path).data
        expect = data
        self.assertEqual(expect, result)

    def test_patterns(self):
        data = {
            "files": []
        }
        with open(self.path, "w") as stream:
            yaml.dump(data, stream)
        config = forest.load_config(self.path)
        result = config.patterns
        expect = []
        self.assertEqual(expect, result)

    def test_patterns(self):
        config = forest.config.Config({
            "files": [{"label": "Name", "pattern": "*.nc"}]
        })
        result = config.patterns
        expect = [("Name", "*.nc")]
        self.assertEqual(expect, result)

    def test_file_groups(self):
        config = forest.config.Config({
            "files": [{"label": "Name", "pattern": "*.nc"}]
        })
        group = config.file_groups[0]
        self.assertEqual(group.label, "Name")
        self.assertEqual(group.pattern, "*.nc")
        self.assertEqual(group.locator, "file_system")
        self.assertEqual(group.file_type, "unified_model")

    def test_file_group_given_directory(self):
        group = forest.config.FileGroup("Name", "*.nc", directory="/dir")
        self.assertEqual(group.directory, "/dir")

    def test_file_group_given_locator(self):
        group = forest.config.FileGroup("Name", "*.nc", locator="database")
        self.assertEqual(group.locator, "database")

    def test_file_group_default_locator(self):
        group = forest.config.FileGroup("Name", "*.nc")
        self.assertEqual(group.locator, "file_system")

    def test_file_group_pattern_given_directory(self):
        group = forest.config.FileGroup("Label", "*.nc", directory="some")
        result = group.full_pattern
        expect = "some/*.nc"
        self.assertEqual(expect, result)


def test_config_parser_given_yaml(tmpdir):
    config_file = str(tmpdir / "test-config.yml")
    content = """
files:
    - label: Hello
      pattern: "*.nc"
    """
    with open(config_file, "w") as stream:
        stream.write(content)
    config = forest.config.Config.load(config_file)
    actual = config.file_groups[0]
    expected = forest.config.FileGroup(
            "Hello", "*.nc",
            locator="file_system",
            directory=None)
    assert actual == expected


def test_config_parser_given_json(tmpdir):
    config_file = str(tmpdir / "test-config.json")
    content = """
"files": [
   {"label": "Hello", "pattern": "*.nc"}
]
    """
    with open(config_file, "w") as stream:
        stream.write(content)
    actual = forest.config.Config.load(config_file)
    assert len(actual.file_groups) == 1
    group = actual.file_groups[0]
    assert group.label == "Hello"
    assert group.pattern == "*.nc"
    assert group.directory is None
    assert group.locator == "file_system"
