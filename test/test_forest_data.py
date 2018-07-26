import unittest
import forest.aws
import forest.data


class TestForestDataset(unittest.TestCase):
    def setUp(self):
        self.file_name = "file.nc"
        self.bucket = forest.aws.SyntheticBucket()
        self.var_name = "var_name"
        self.var_lookup = {
            self.var_name: {
                "stash_section": None,
                "stash_item": None
            }
        }
        self.dataset = forest.data.ForestDataset(self.file_name,
                                                 self.bucket,
                                                 self.var_lookup)

    def test_can_be_constructed(self):
        var_lookup = None
        forest.data.ForestDataset(self.file_name,
                                  self.bucket,
                                  var_lookup)

    def test_get_var_lookup_mslp(self):
        config = forest.data.GA6_CONF_ID
        var_lookup = forest.data.get_var_lookup(config)
        result = var_lookup['mslp']
        expect = {
            "accumulate": False,
            "filename": "umnsaa_pverb",
            "stash_item": 222,
            "stash_name": "air_pressure_at_sea_level",
            "stash_section": 16
        }
        self.assertEqual(result, expect)
