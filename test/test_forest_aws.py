"""Amazon web services infrastructure"""
import unittest
import unittest.mock
import os
import sys
import forest.aws
import forest.data


@unittest.mock.patch("forest.aws.urllib")
@unittest.mock.patch("forest.aws.os")
@unittest.mock.patch("forest.aws.print", autospec=True)
class TestDownloadFromS3(unittest.TestCase):
    def test_download_from_s3_calls_urlretrieve(self,
                                                mock_print,
                                                os,
                                                urllib):
        s3_url = "s3_url"
        local_path = "local_path"
        os.path.isfile.return_value = False
        forest.aws.S3Bucket.s3_download(s3_url, local_path)
        urllib.request.urlretrieve.assert_called_once_with(s3_url,
                                                           local_path)


class TestS3Mount(unittest.TestCase):
    def setUp(self):
        self.file_name = "file.nc"
        self.directory = "directory"
        self.mount = forest.aws.S3Mount(self.directory)

    def test_mount_directory(self):
        self.assertEqual(self.mount.directory, self.directory)

    @unittest.mock.patch("forest.aws.os")
    def test_file_exists_queries_file_system(self, os):
        self.mount.file_exists(self.file_name)
        os.path.isfile.assert_called_once_with(self.file_name)

    def test_path_to_load(self):
        result = self.mount.path_to_load(self.file_name)
        expect = os.path.join(self.directory, self.file_name)
        self.assertEqual(result, expect)


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
        self.download_dir = "download_directory"
        self.bucket = forest.aws.S3Bucket(self.server_address,
                                          self.bucket_name,
                                          self.download_dir)

    def test_server_address(self):
        self.assertEqual(self.bucket.server_address, self.server_address)

    def test_bucket_name(self):
        self.assertEqual(self.bucket.bucket_name, self.bucket_name)

    def test_path_to_load(self):
        result = self.bucket.path_to_load(self.file_name)
        expect = os.path.join(self.download_dir, self.file_name)
        self.assertEqual(result, expect)

@unittest.mock.patch("forest.aws.os")
@unittest.mock.patch("forest.aws.urllib")
@unittest.mock.patch("forest.aws.print", autospec=True)
class TestS3BucketIO(unittest.TestCase):
    def setUp(self):
        self.file_name = "file.nc"
        self.server_address = "server_address"
        self.bucket_name = "bucket_name"
        self.download_dir = "download_directory"
        self.bucket = forest.aws.S3Bucket(self.server_address,
                                          self.bucket_name,
                                          self.download_dir)

    def test_file_exists_calls_s3_file_exists(self,
                                              mock_print,
                                              urllib,
                                              os):
        self.bucket.s3_file_exists = unittest.mock.Mock()
        self.bucket.file_exists(self.file_name)
        expect = self.bucket.s3_url(self.file_name)
        self.bucket.s3_file_exists.assert_called_once_with(expect)

    def test_local_file_exists_calls_isfile(self,
                                            mock_print,
                                            urllib,
                                            os):
        self.bucket.local_file_exists(self.file_name)
        expect = self.bucket.path_to_load(self.file_name)
        os.path.isfile.assert_called_once_with(expect)

    def test_load_file_calls_makedirs(self,
                                      mock_print,
                                      urllib,
                                      os):
        os.path.isdir.return_value = False
        self.bucket.load_file(self.file_name)
        os.path.isdir.assert_called_once_with(self.download_dir)
        os.makedirs.assert_called_once_with(self.download_dir)

    def test_load_file_calls_s3_download(self,
                                         mock_print,
                                         urllib,
                                         os):
        self.bucket.local_file_exists = unittest.mock.Mock(return_value=False)
        self.bucket.remote_file_exists = unittest.mock.Mock(return_value=True)
        self.bucket.s3_download = unittest.mock.Mock()
        self.bucket.load_file(self.file_name)
        url = self.bucket.s3_url(self.file_name)
        local_file = self.bucket.path_to_load(self.file_name)
        self.bucket.s3_download.assert_called_once_with(url, local_file)
