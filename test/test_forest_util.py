import unittest
import forest


class TestDownloadFromS3(unittest.TestCase):
    @unittest.mock.patch("forest.util.urllib")
    @unittest.mock.patch("forest.util.os")
    def test_download_from_s3_calls_urlretrieve(self, os, urllib):
        s3_url = "s3_url"
        local_path = "local_path"
        os.path.isfile.return_value = False
        forest.util.download_from_s3(s3_url, local_path)
        urllib.request.urlretrieve.assert_called_once_with(s3_url,
                                                           local_path)
