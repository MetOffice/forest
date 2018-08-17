import unittest
import datetime as dt
import forest.aws
import forest.data


class TestGetAvailableDatasets(unittest.TestCase):
    @unittest.skip("going down the rabbit hole")
    def test_get_available_datasets(self):
        file_loader = None
        dataset_template = None
        days_since_period_start = None
        num_days = None
        model_period = None
        forest.data.get_available_datasets(file_loader,
                                           dataset_template,
                                           days_since_period_start,
                                           num_days,
                                           model_period)

    def test_get_model_run_times(self):
        days_since_period_start = 0
        num_days = 1
        model_period = 24
        result = forest.data.get_model_run_times(days_since_period_start,
                                                 num_days,
                                                 model_period)
        expect = [dt.datetime(2018, 8, 17, tzinfo=dt.timezone.utc)], ['20180817T0000Z']
        self.assertEqual(result, expect)


class TestForestDataset(unittest.TestCase):
    def setUp(self):
        self.file_name = "file.nc"
        self.bucket = forest.aws.S3Mount("directory")
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
