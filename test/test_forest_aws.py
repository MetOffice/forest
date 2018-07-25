"""Amazon web services infrastructure"""
import unittest
import unittest.mock
import os
import sys
import forest.aws
import forest.data


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
class TestS3BucketIO(unittest.TestCase):
    def setUp(self):
        self.file_name = "file.nc"
        self.server_address = "server_address"
        self.bucket_name = "bucket_name"
        self.bucket = forest.aws.S3Bucket(self.server_address,
                                          self.bucket_name)

    def test_file_exists_calls_remote_file_exists(self, util, os):
        self.bucket.do_download = True
        self.bucket.file_exists(self.file_name)
        expect = self.bucket.s3_url(self.file_name)
        util.check_remote_file_exists.assert_called_once_with(expect)

    def test_file_exists_calls_isfile(self, util, os):
        self.bucket.do_download = False
        self.bucket.file_exists(self.file_name)
        expect = self.bucket.path_to_load(self.file_name)
        os.path.isfile.assert_called_once_with(expect)

    def test_retrieve_file_calls_makedirs(self, util, os):
        os.path.isdir.return_value = False
        self.bucket.retrieve_file(self.file_name, verbose=False)
        directory = self.bucket.base_path_local
        os.path.isdir.assert_called_once_with(directory)
        os.makedirs.assert_called_once_with(directory)

    def test_retrieve_file_calls_download_from_s3(self, util, os):
        self.bucket.retrieve_file(self.file_name)
        directory = self.bucket.base_path_local
        url = self.bucket.s3_url(self.file_name)
        local_file = self.bucket.local_path(self.file_name)
        util.download_from_s3.assert_called_once_with(url, local_file)


class TestSyntheticBucket(unittest.TestCase):
    """unit test synthetic data"""
    def setUp(self):
        self.file_name = None
        self.constraint = None
        self.samples = forest.aws.SyntheticBucket()
        self.cube = self.samples.load_cube(self.file_name, self.constraint)

    def test_path_to_load_function_exists(self):
        self.samples.path_to_load(self.file_name)

    def test_load_cube_has_time_dimension(self):
        self.cube.coord('time').points

    def test_load_cube_has_at_least_two_time_points(self):
        self.cube.coord('time').points[1]

    def test_load_cube_returns_cube_with_latitude_dimcoord(self):
        self.cube.coords('latitude')[0].points

    def test_load_cube_returns_cube_with_longitude_dimcoord(self):
        self.cube.coords('longitude')[0].points

    def test_load_cube_has_attributes_stash_section(self):
        self.cube.attributes['STASH'].section

    @unittest.skip("understanding ForestDataset")
    def test_get_available_times(self):
        """sample data should satisfy forest.data.get_available_times"""
        file_name = None
        bucket = self.samples
        variable = "precipitation"  # Hard-coded Forest key
        var_lookup = {
            variable: {
                "stash_section": "section"
            }
        }
        datasets = {
            "name": {
                "data": forest.data.ForestDataset(file_name,
                                                  bucket,
                                                  var_lookup)
            }
        }
        forest.data.get_available_times(datasets, variable)
