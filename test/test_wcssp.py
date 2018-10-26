import unittest
import unittest.mock
import datetime as dt
import yaml
import os
import sys
from forest.test.util import remove_after
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../bokeh_apps"))
import wcssp.main


class TestParseEnvironment(unittest.TestCase):
    """Similar idea to argparse.ArgumentParser.parse_args()

    Bokeh applications do not accept command line arguments
    so an app must be configured by environment variables
    """
    def test_parse_environment_download_data_default(self):
        env = {}
        self.check_parse_environment(env, "download_data", False)

    def test_parse_environment_download_data_true(self):
        env = {"FOREST_DOWNLOAD_DATA": "True"}
        self.check_parse_environment(env, "download_data", True)

    def test_parse_environment_download_data_false(self):
        env = {"FOREST_DOWNLOAD_DATA": "FALSE"}
        self.check_parse_environment(env, "download_data", False)

    def test_parse_environment_download_directory_default(self):
        env = {}
        self.check_parse_environment(env, "download_directory",
                                     os.path.expanduser("~/SEA_data/"))

    def test_parse_environment_download_directory_given_local_root(self):
        env = {"LOCAL_ROOT": "/dir"}
        self.check_parse_environment(env, "download_directory",
                                     os.path.expanduser("/dir"))

    def test_parse_environment_download_directory_given_download_dir(self):
        env = {"FOREST_DOWNLOAD_DIR": "/dir"}
        self.check_parse_environment(env, "download_directory",
                                     os.path.expanduser("/dir"))

    def test_parse_environment_mount_directory(self):
        env = {"S3_ROOT": "/dir"}
        expect = os.path.expanduser("/dir/stephen-sea-public-london")
        self.check_parse_environment(env, "mount_directory", expect)

    def test_parse_environment_mount_directory(self):
        env = {"FOREST_MOUNT_DIR": "/mount/dir"}
        expect = os.path.expanduser("/mount/dir")
        self.check_parse_environment(env, "mount_directory", expect)

    def test_parse_environment_start_date_default(self):
        env = {}
        expect = (dt.datetime.now() - dt.timedelta(days=7)).replace(second=0,
                                                                    microsecond=0)
        self.check_parse_environment(env, "start_date", expect)

    def test_parse_environment_start_date_given_forest_start(self):
        env = {"FOREST_START": "20180101"}
        expect = dt.datetime(2018, 1, 1)
        self.check_parse_environment(env, "start_date", expect)

    def test_parse_config_file(self):
        self.check_parse_environment({}, "config_file", None)

    def test_parse_config_file_given_forest_config_file(self):
        given = {"FOREST_CONFIG_FILE": "file.cfg"}
        self.check_parse_environment(given, "config_file", "file.cfg")

    def check_parse_environment(self, env, attr, expect):
        args = wcssp.main.parse_environment(env)
        result = getattr(args, attr)
        self.assertEqual(expect, result)


class TestLoadConfig(unittest.TestCase):
    def test_load_config(self):
        file_name = remove_after(self, "test-load-environment.yaml")
        settings = {
            "models": [
                {
                    "name": "East Africa 4.4km",
                    "file_pattern": "eakm4p4_{run_date:%Y%m%dT%H%MZ}.nc"
                }
            ]
        }
        with open(file_name, "w") as stream:
            yaml.dump(settings, stream)
        result = wcssp.main.load_config(file_name)
        expect = settings
        self.assertEqual(result, expect)


class TestSouthEastAsiaConfig(unittest.TestCase):
    def setUp(self):
        self.models = [
            {
                "name": "N1280 GA6 LAM Model",
                "file": {
                    "pattern": "SEA_n1280_ga6_{:%Y%m%dT%H%MZ}.nc",
                    "format": "ga6"
                }
            },
            {
                "name": "SE Asia 4.4KM RA1-T ",
                "file": {
                    "pattern": "SEA_km4p4_ra1t_{:%Y%m%dT%H%MZ}.nc",
                    "format": "ra1t"
                }
            },
            {
                "name": "Indonesia 1.5KM RA1-T",
                "file": {
                    "pattern": "SEA_indon2km1p5_ra1t_{:%Y%m%dT%H%MZ}.nc",
                    "format": "ra1t"
                }
            },
            {
                "name": "Malaysia 1.5KM RA1-T",
                "file": {
                    "pattern": "SEA_mal2km1p5_ra1t_{:%Y%m%dT%H%MZ}.nc",
                    "format": "ra1t"
                }
            },
            {
                "name": "Philipines 1.5KM RA1-T",
                "file": {
                    "pattern": "SEA_phi2km1p5_ra1t_{:%Y%m%dT%H%MZ}.nc",
                    "format": "ra1t"
                }
            }
        ]
        self.regions = [
            {
                "name": "South east Asia",
                "longitude_range": [90.0, 153.96],
                "latitude_range": [-18.0, 29.96],
            },
            {
                "name": "Indonesia",
                "longitude_range": [99.875, 120.111],
                "latitude_range": [-15.1, 1.0865]
            },
            {
                "name": "Malaysia",
                "longitude_range": [95.25, 108.737],
                "latitude_range": [-2.75, 10.7365]
            },
            {
                "name": "Philippines",
                "longitude_range": [115.8, 131.987],
                "latitude_range": [3.1375, 21.349]
            },
        ]
        self.maxDiff = None
        self.config = wcssp.main.south_east_asia_config()

    def test_south_east_asia_regions(self):
        result = self.config["regions"]
        expect = self.regions
        self.assertEqual(result, expect)

    def test_south_east_asia_config_models(self):
        result = self.config["models"]
        expect = self.models
        self.assertEqual(result, expect)

    def test_south_east_asia_config_title(self):
        result = self.config["title"]
        expect = "Two model comparison"
        self.assertEqual(result, expect)
