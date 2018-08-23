import unittest
import os
import datetime as dt
import forest.aws
import forest.data


class FakeLoader(object):
    def file_exists(self, path):
        return True


class TestFormat(unittest.TestCase):
    def test_format(self):
        config = "config"
        result = "{}_{{:%Y%m%d}}.nc".format(config)
        expect = "config_{:%Y%m%d}.nc"
        self.assertEqual(result, expect)


class TestGetAvailableDatasets(unittest.TestCase):
    @unittest.skip("too complicated to test")
    def test_get_available_datasets(self):
        file_loader = None
        dataset_template = {}
        model_run_times = [dt.datetime(2018, 8, 17, tzinfo=dt.timezone.utc)]
        result = forest.data.get_available_datasets(file_loader,
                                                    dataset_template,
                                                    model_run_times)
        expect = {'20180817T0000Z': {}}
        self.assertEqual(result, expect)

    def test_get_model_run_times(self):
        period_start = dt.datetime(2018, 8, 17)
        num_days = 1
        model_period = 24
        result = forest.data.get_model_run_times(period_start,
                                                 num_days,
                                                 model_period)
        expect = [dt.datetime(2018, 8, 17, tzinfo=dt.timezone.utc)]
        self.assertEqual(result, expect)

    def test_format_model_run_time(self):
        model_run_time = dt.datetime(2018, 8, 17, tzinfo=dt.timezone.utc)
        result = forest.data.format_model_run_time(model_run_time)
        expect = '20180817T0000Z'
        self.assertEqual(result, expect)

    def test_get_var_lookup_ga6_keys(self):
        config = "ga6"
        lookup = forest.data.get_var_lookup(config)
        result = sorted(lookup.keys())
        expect = [
            'air_temperature',
            'cloud_fraction',
            'mslp',
            'precipitation',
            'relative_humidity',
            'wet_bulb_potential_temp',
            'x_wind',
            'x_winds_upper',
            'y_wind',
            'y_winds_upper'
        ]
        self.assertEqual(result, expect)

    def test_config_file_given_ga6_returns_existing_file(self):
        config = "ga6"
        result = forest.data.config_file(config)
        self.assertTrue(os.path.exists(result))


class TestForestDataset(unittest.TestCase):
    def setUp(self):
        self.test_directory = os.path.dirname(os.path.realpath(__file__))
        self.file_name = "SEA_phi2km1p5_ra1t_20180821T0000Z.nc"
        self.bucket = forest.aws.S3Mount(self.test_directory)

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

    def test_get_data_should_support_model_run_time(self):
        var_lookup = forest.data.get_var_lookup("ra1t")
        dataset = forest.data.ForestDataset(self.file_name,
                                            self.bucket,
                                            var_lookup)
        variable = "precipitation"
        selected_time = dt.datetime(2018, 1, 1)
        dataset.get_data(variable, selected_time)
        self.assertTrue(False)
