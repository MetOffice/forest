import unittest
import unittest.mock
import forest
import numpy as np

class TestForestDataset(unittest.TestCase):
    """Test suite to understand ForestDataset class"""
    def test_can_be_constructed(self):
        forest.data.ForestDataset(*self.generic_args())

    @unittest.skip("testing load_data")
    def test_get_data(self):
        var_name = 'air_temperature'
        selected_time = None
        dataset = forest.data.ForestDataset(*self.generic_args())
        dataset.get_data(var_name, selected_time)

    @unittest.skip("testing basic_cube_load")
    def test_load_data(self):
        var_name = 'air_temperature'
        time_ix = 0
        dataset = forest.data.ForestDataset(*self.generic_args(
            var_lookup = {
                var_name: None
            }
        ))
        dataset.load_data(var_name, time_ix)

    @unittest.mock.patch("forest.data.iris")
    def test_basic_cube_load_calls_load_cube(self, iris):
        """Load an iris cube"""
        iris.__version__ = '2.0.0'
        constraint = iris.Constraint.return_value
        var_name = 'air_temperature'
        time_ix = 0
        field_dict = {
            'stash_section': None,
            'stash_item': None
        }
        dataset = forest.data.ForestDataset(*self.generic_args(
            var_lookup = {
                var_name: field_dict
            }
        ))
        dataset.data[var_name] = {
        }
        dataset.basic_cube_load(var_name, time_ix)
        iris.load_cube.assert_called_once_with(dataset.path_to_load,
                                               constraint)

    @unittest.mock.patch("forest.data.iris")
    def test_basic_cube_load_returns_cube(self, iris):
        iris.__version__ = '2.0.0'
        var_name = 'air_temperature'
        time_ix = 0
        field_dict = {
            'stash_section': None,
            'stash_item': None
        }
        dataset = forest.data.ForestDataset(*self.generic_args(
            var_lookup = {
                var_name: field_dict
            }
        ))
        dataset.data[var_name] = {
        }
        dataset.basic_cube_load(var_name, time_ix)
        self.assertEqual(dataset.data[var_name][time_ix],
                         iris.load_cube.return_value)

    def generic_args(self, var_lookup=None):
        config = None
        file_name = "file_name"
        s3_base = "s3_base"
        s3_local_base = "s3_local_base"
        use_s3_mount = None
        base_local_path = "base_local_path"
        do_download = None
        return (config, file_name, s3_base, s3_local_base, use_s3_mount, base_local_path, do_download, var_lookup)

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
