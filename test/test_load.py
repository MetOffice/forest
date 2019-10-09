import yaml
import forest
from forest import main


def test_earth_networks_loader_given_pattern():
    loader = forest.Loader.from_pattern("Label", "EarthNetworks*.txt", "earth_networks")
    assert isinstance(loader, forest.earth_networks.Loader)


def test_build_loader_given_files():
    """replicate main.py as close as possible"""
    files = ["file_20190101T0000Z.nc"]
    args = main.parse_args.parse_args(files)
    config = forest.config.from_files(args.files, args.file_type)
    group = config.file_groups[0]
    loader = forest.Loader.group_args(group, args)
    assert isinstance(loader, forest.data.DBLoader)
    assert loader.locator.paths == files


def test_build_loader_given_database(tmpdir):
    """replicate main.py as close as possible"""
    database_file = str(tmpdir / "database.db")

    config_file = str(tmpdir / "config.yml")
    settings = {
        "files": [
            {
                "label": "UM",
                "pattern": "*.nc",
                "locator": "database"
            }
        ]
    }
    with open(config_file, "w") as stream:
        yaml.dump(settings, stream)

    args = main.parse_args.parse_args([
        "--database", database_file,
        "--config-file", config_file])
    config = forest.config.load_config(args.config_file)
    group = config.file_groups[0]
    database = forest.db.Database.connect(database_file)
    loader = forest.Loader.group_args(group, args, database=database)
    database.close()
    assert hasattr(loader.locator, "connection")
    assert loader.locator.directory is None


def test_build_loader_given_database_and_directory(tmpdir):
    database_file = str(tmpdir / "database.db")
    config_file = str(tmpdir / "config.yml")
    args = main.parse_args.parse_args([
        "--database", database_file,
        "--config-file", config_file])
    label = "UM"
    pattern = "*.nc"
    directory = "/some/dir"
    group = forest.config.FileGroup(
            label,
            pattern,
            directory=directory,
            locator="database")
    database = forest.db.Database.connect(database_file)
    loader = forest.Loader.group_args(group, args, database=database)
    database.close()
    assert hasattr(loader.locator, "connection")
    assert loader.locator.directory == directory


def test_build_loader_given_config_file_pattern(tmpdir):
    config_file = str(tmpdir / "config.yml")
    path = str(tmpdir / "file_20190101T0000Z.nc")
    with open(path, "w"):
        pass
    args = main.parse_args.parse_args([
        "--config-file", config_file])
    label = "UM"
    pattern = "*.nc"
    directory = str(tmpdir)
    group = forest.config.FileGroup(
            label,
            pattern,
            directory=directory,
            locator="file_system")
    loader = forest.Loader.group_args(group, args)
    assert loader.locator.paths == [path]


def test_build_loader_given_eida50_file_type():
    label = "EIDA50"
    pattern = "eida50*.nc"
    file_type = "eida50"
    loader = forest.Loader.from_pattern(label, pattern, file_type)
    assert isinstance(loader, forest.satellite.EIDA50)
    assert isinstance(loader.locator, forest.satellite.Locator)


def test_build_loader_given_rdt_file_type():
    loader = forest.Loader.from_pattern(
            "Label", "*.json", "rdt")
    assert isinstance(loader, forest.rdt.Loader)
    assert isinstance(loader.locator, forest.rdt.Locator)


def test_replace_dir_given_args_dir_only():
    check_replace_dir("args/dir", None, "args/dir")


def test_replace_dir_given_group_dir_only():
    check_replace_dir(None, "group/dir", "group/dir")


def test_replace_dir_given_relative_group_dir_appends_to_args_dir():
    check_replace_dir("args/dir", "leaf", "args/dir/leaf")


def test_replace_dir_given_absolute_group_dir_overrides_rel_args_dir():
    check_replace_dir("args/relative", "/group/absolute", "/group/absolute")


def test_replace_dir_given_absolute_group_dir_overrides_abs_args_dir():
    check_replace_dir("/args/absolute", "/group/absolute", "/group/absolute")


def check_replace_dir(args_dir, group_dir, expected):
    actual = forest.Loader.replace_dir(args_dir, group_dir)
    assert actual == expected


def test_full_pattern_given_name_only():
    check_full_pattern("file.nc", None, None, "file.nc")


def test_full_pattern_given_relative_prefix_dir():
    check_full_pattern("file.nc", None, "prefix", "prefix/file.nc")


def test_full_pattern_given_relative_leaf_and_prefix_dir():
    check_full_pattern("file.nc", "leaf", "prefix", "prefix/leaf/file.nc")


def test_full_pattern_given_absolute_leaf_ignores_relative_prefix():
    check_full_pattern("file.nc", "/leaf", "prefix", "/leaf/file.nc")


def test_full_pattern_given_absolute_leaf_ignores_absolute_prefix():
    check_full_pattern("file.nc", "/leaf", "/prefix", "/leaf/file.nc")


def check_full_pattern(name, leaf, prefix, expected):
    actual = forest.Loader.full_pattern(name, leaf, prefix)
    assert actual == expected
