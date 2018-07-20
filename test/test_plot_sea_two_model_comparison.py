import unittest
import unittest.mock
import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../bokeh_apps/plot_sea_two_model_comparison"))
import main
import forest


class TestPlotSeaTwoModelComparison(unittest.TestCase):
    @unittest.skip("system not decoupled yet")
    def test_main(self, bokeh):
        bokeh_id = "bokeh_id"
        main.main(bokeh_id)

    @unittest.skip("unit test dataset")
    def test_get_available_datasets(self):
        forest.data.get_available_datasets()


@unittest.skip("deprecated API")
class TestForestDataset(unittest.TestCase):
    def test_can_be_constructed(self):
        forest.data.ForestDataset(*self.args())

    @unittest.skip("understanding var_lookup")
    def test_get_data(self):
        var_name = 'precipitation'
        selected_time = None
        dataset = forest.data.ForestDataset(*self.args())
        dataset.check_data = unittest.mock.Mock(return_value=True)
        cube = dataset.get_data(var_name,
                                selected_time)

    def test_check_data(self):
        dataset = forest.data.ForestDataset(*self.args())
        self.assertEqual(dataset.check_data(), False)

    def test_retrieve_data(self):
        dataset = forest.data.ForestDataset(*self.args())
        dataset.retrieve_data()

    @unittest.skip("understanding var_lookup")
    def test_load_times(self):
        var_name = 'precipitation'
        dataset = forest.data.ForestDataset(*self.args())
        dataset.load_times(var_name)

    def test_get_var_lookup(self):
        lookup = forest.data.get_var_lookup(forest.data.GA6_CONF_ID)
        result = lookup['precipitation']
        expect = {
            'accumulate': True,
            'filename': 'umnsaa_pverb',
            'stash_item': 216,
            'stash_name': 'precipitation_flux',
            'stash_section': 5
        }
        self.maxDiff = None
        self.assertEqual(expect, result)

    def test_ga6_conf_id(self):
        self.assertEqual(forest.data.GA6_CONF_ID, "ga6")

    def args(self):
        config = None
        file_name = "file_name"
        s3_base = "s3_base"
        s3_local_base = "s3_local_base"
        use_s3_mount = None
        base_local_path = "base_local_path"
        do_download = None
        var_lookup = None
        return (config,
                file_name,
                s3_base,
                s3_local_base,
                use_s3_mount,
                base_local_path,
                do_download,
                var_lookup)
