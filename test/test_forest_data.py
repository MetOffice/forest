import unittest
import forest.aws
import forest.data


class TestForestDataset(unittest.TestCase):
    def test_can_be_constructed(self):
        file_name = "file_name"
        bucket = forest.aws.S3Bucket("server", "bucket")
        var_lookup = None
        forest.data.ForestDataset(file_name,
                                  bucket,
                                  var_lookup)
