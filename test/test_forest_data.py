import unittest
import forest
import numpy as np

class TestForestDataset(unittest.TestCase):
    def test_can_be_constructed(self):
        forest.data.ForestDataset(*self.generic_args())

    def generic_args(self):
        config = None
        file_name = None
        s3_base = None
        s3_base_local = None
        use_s3_mount = None
        base_local_path = None
        do_download = None
        var_lookup = None
        return (config, file_name, s3_base, s3_base_local, use_s3_mount, base_local_path, do_download, var_lookup)

class TestGetAvailableTimes(unittest.TestCase):
    def test_get_available_times(self):
        mock_dataset = unittest.mock.Mock()
        mock_dataset.get_times.return_value = []
        datasets = {
            "key": {
                "data": mock_dataset
            }
        }
        variable = "variable"
        result = forest.data.get_available_times(datasets, variable)
        expect = []
        np.testing.assert_array_equal(expect, result)
