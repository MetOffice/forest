"""Amazon web services infrastructure"""
import unittest
import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../bokeh_apps",
                                "plot_sea_two_model_comparison"))
import main


class TestS3Bucket(unittest.TestCase):
    def setUp(self):
        self.server_address = "https://s3.eu-west-2.amazonaws.com"
        self.bucket_name = "stephen-sea-public-london"
        self.s3_base = "{server}/{bucket}/model_data/".format(server=self.server_address,
                                                              bucket=self.bucket_name)
        self.bucket = main.S3Bucket()

    def tearDown(self):
        if "S3_ROOT" in os.environ:
            del os.environ["S3_ROOT"]

    def test_server_address(self):
        self.assertEqual(self.bucket.server_address, self.server_address)

    def test_bucket_name(self):
        self.assertEqual(self.bucket.bucket_name, self.bucket_name)

    def test_s3_base(self):
        self.assertEqual(self.bucket.s3_base, self.s3_base)

    def test_s3_root_given_environment_variable(self):
        s3_root_variable = "s3_root_variable"
        os.environ["S3_ROOT"] = s3_root_variable
        self.assertEqual(self.bucket.s3_root, s3_root_variable)

    def test_s3_root_given_user_directory(self):
        self.assertEqual(self.bucket.s3_root, os.path.expanduser("~/s3"))
