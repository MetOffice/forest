import unittest
import unittest.mock
import datetime as dt
import yaml
import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../bokeh_apps"))
import plot_sea_two_model_comparison.main


class TestPlotSeaTwoModelComparison(unittest.TestCase):
    """Need to develop appropriate testing strategy

    I would like to be able to run the main program with
    a minimal set of sub-systems mocked
    """
    def setUp(self):
        self.bokeh_id = "bk-id"

    @unittest.skip("too difficult to test")
    def test_plot_sea_two_model_comparison(self):
        plot_sea_two_model_comparison.main.main(self.bokeh_id)


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
        args = plot_sea_two_model_comparison.main.parse_environment(env)
        result = getattr(args, attr)
        self.assertEqual(expect, result)


class TestLoadConfig(unittest.TestCase):
    def test_load_config(self):
        file_name = "test-load-environment.yaml"
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
        result = plot_sea_two_model_comparison.main.load_config(file_name)
        expect = settings
        self.assertEqual(result, expect)

    def test_south_east_asia_config(self):
        result = plot_sea_two_model_comparison.main.south_east_asia_config()
        expect = {
            "models": [
                {
                    "name": "N1280 GA6 LAM Model",
                    "file_pattern": "SEA_n1280_ga6_{%Y%m%dT%H%MZ}.nc"
                },
                {
                    "name": "SE Asia 4.4KM RA1-T ",
                    "file_pattern": "SEA_km4p4_ra1t_{%Y%m%dT%H%MZ}.nc"
                },
                {
                    "name": "Indonesia 1.5KM RA1-T",
                    "file_pattern": "SEA_indon2km1p5_{%Y%m%dT%H%MZ}.nc"
                },
                {
                    "name": "Malaysia 1.5KM RA1-T",
                    "file_pattern": "SEA_mal2km1p5_{%Y%m%dT%H%MZ}.nc"
                },
                {
                    "name": "Philipines 1.5KM RA1-T",
                    "file_pattern": "SEA_phi2km1p5_{%Y%m%dT%H%MZ}.nc"
                }
            ]
        }
        self.maxDiff = None
        self.assertEqual(result, expect)
