import pytest
import unittest
import pytest
import yaml
import os
import forest


SERVER_CONFIG = os.path.join(os.path.dirname(__file__),
                             '../server/config-philippines.yaml')


@pytest.mark.parametrize("label,settings", [
    ("GA7", {"pattern": "*ga7*.nc"}),
])
def test_server_config(label, settings):
    with open(SERVER_CONFIG) as stream:
        data = yaml.load(stream)
    labels = [ds["label"] for ds in data["files"]]
    assert label in labels
    for dataset in data["files"]:
        if dataset["label"] == label:
            for key, value in settings.items():
                assert dataset[key] == value


@pytest.mark.parametrize("env,args,expected", [
        ({"env": "variable"}, None, {"env": "variable"}),
        ({}, {}, {}),
        ({}, [], {}),
        ({}, {"k": "v"}, {"k": "v"}),
        ({"k": "v"}, {}, {"k": "v"}),
        ({"x": "environment"}, {"x": "user"}, {"x": "user"}),
        ({"x": "environment"}, [("x", "a"), ("y", "b")], {"x": "a", "y": "b"}),
        ({"z": "c"}, [["x", "a"], ["y", "b"]], {"x": "a", "y": "b", "z": "c"}),
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
                directory="${BUCKET_DIR}/unified_model")
        self.assert_group_equal(expect, result)

    def test_load_server_config_second_group(self):
        result = self.config.file_groups[2]
        expect = forest.config.FileGroup(
                "Operational Tropical Africa",
                "*os42_ea*.nc",
                locator="database",
                directory="${BUCKET_DIR}/unified_model")
        self.assert_group_equal(expect, result)

    def test_load_server_config_has_eida50(self):
        groups = [g for g in self.config.file_groups
            if g.file_type == 'eida50']
        result = groups[0]
        expect = forest.config.FileGroup(
                "EIDA50",
                "${BUCKET_DIR}/eida50/EIDA50_takm4p4*.nc",
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
                 "pattern": "~/cache/*.nc"}
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

    def test_file_group_given_locator(self):
        group = forest.config.FileGroup("Name", "*.nc", locator="database")
        self.assertEqual(group.locator, "database")

    def test_file_group_default_locator(self):
        group = forest.config.FileGroup("Name", "*.nc")
        self.assertEqual(group.locator, "file_system")


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
            locator="file_system")
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
    assert group.locator == "file_system"


@pytest.mark.parametrize("data,expect", [
    ({}, None),
    ({"presets": {}}, None),
    ({"presets": {"file": "/some.json"}}, "/some.json")
])
def test_config_parser_presets_file(data, expect):
    config = forest.config.Config(data)
    assert config.presets_file == expect


@pytest.mark.parametrize("data,expect", [
    ({}, True),
    ({"use_web_map_tiles": False}, False),
])
def test_config_parser_use_web_map_tiles(data, expect):
    config = forest.config.Config(data)
    assert config.use_web_map_tiles == expect


@pytest.mark.parametrize("data,expect", [
    ({}, False),
    ({"features": {"example": True}}, True),
])
def test_config_parser_features(data, expect):
    config = forest.config.Config(data)
    assert config.features["example"] == expect


def test_config_parser_plugin_entry_points():
    config = forest.config.Config({
        "plugins": {
            "feature": {
                "entry_point": "module.main"
            }
        }
    })
    assert config.plugins["feature"].entry_point == "module.main"


def test_config_parser_plugin_given_unsupported_key():
    with pytest.raises(Exception):
         forest.config.Config({
            "plugins": {
                "not_a_key": {
                    "entry_point": "module.main"
                }
            }
        })


def test_config_default_state():
    config = forest.config.Config({
        "state": {}
    })
    assert config.state == forest.state.State.from_dict({})
