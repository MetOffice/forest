import unittest
import unittest.mock
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

    def test_plot_sea_two_model_comparison(self):
        module = "plot_sea_two_model_comparison.main.forest.data.get_available_datasets"
        with unittest.mock.patch(module):
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

    def check_parse_environment(self, env, attr, expect):
        args = plot_sea_two_model_comparison.main.parse_environment(env)
        result = getattr(args, attr)
        self.assertEqual(expect, result)
