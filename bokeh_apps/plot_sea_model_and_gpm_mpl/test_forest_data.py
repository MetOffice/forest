import unittest
import forest
import numpy as np


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
