"""Amazon web services infrastructure"""
import unittest
import unittest.mock
import os
import sys
import forest.aws


class TestS3Bucket(unittest.TestCase):
    """AWS S3 architecture

    Forest application should run from either file system
    or AWS download seemlessly. Both systems should present
    a simple API to the internals of Forest
    """
    def setUp(self):
        self.file_name = "file.nc"
        self.server_address = "https://s3.eu-west-2.amazonaws.com"
        self.bucket_name = "stephen-sea-public-london"
        self.s3_base = "{server}/{bucket}/model_data/".format(server=self.server_address,
                                                              bucket=self.bucket_name)
        self.bucket = forest.aws.S3Bucket(self.server_address,
                                          self.bucket_name)

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

    def test_s3_local_base(self):
        expect = "{}/{}/model_data".format(self.bucket.s3_root,
                                           self.bucket.bucket_name)
        self.assertEqual(self.bucket.s3_local_base, expect)

    def test_use_s3_mount_returns_true(self):
        self.assertEqual(self.bucket.use_s3_mount, True)

    def test_do_download_returns_false(self):
        self.assertEqual(self.bucket.do_download, False)

    def test_base_path_local(self):
        self.assertEqual(self.bucket.base_path_local, os.path.expanduser("~/SEA_data/model_data"))

    def test_path_to_load_given_use_s3_mount_true(self):
        self.bucket.use_s3_mount = True
        self.assertEqual(self.bucket.path_to_load(self.file_name),
                         self.bucket.s3_local_path(self.file_name))

    def test_path_to_load_given_use_s3_mount_false(self):
        self.bucket.use_s3_mount = False
        self.assertEqual(self.bucket.path_to_load(self.file_name),
                         self.bucket.local_path(self.file_name))

    @unittest.mock.patch("forest.aws.os")
    @unittest.mock.patch("forest.aws.util")
    def test_file_exists_calls_remote_file_exists(self, util, os):
        self.bucket.do_download = True
        self.bucket.file_exists(self.file_name)
        expect = self.bucket.s3_url(self.file_name)
        util.check_remote_file_exists.assert_called_once_with(expect)

    @unittest.mock.patch("forest.aws.os")
    @unittest.mock.patch("forest.aws.util")
    def test_file_exists_calls_isfile(self, util, os):
        self.bucket.do_download = False
        self.bucket.file_exists(self.file_name)
        expect = self.bucket.path_to_load(self.file_name)
        os.path.isfile.assert_called_once_with(expect)
