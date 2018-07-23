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

    @unittest.skip("testing basic_cube_load")
    def test_dataset_get_times(self):
        var_name = "var_name"
        var_lookup = {
            var_name: None
        }
        dataset = forest.data.ForestDataset(self.file_name,
                                            self.bucket,
                                            var_lookup)
        dataset.get_times(var_name)

    def test_init_makes_data_dictionary(self):
        self.assertEqual(self.dataset.data, {})

    def test_basic_cube_load(self):
        time_ix = 0
        dataset.basic_cube_load(var_name, time_ix)
