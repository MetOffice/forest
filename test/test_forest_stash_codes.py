import unittest
import forest


class TestForestStashCodes(unittest.TestCase):
    def setUp(self):
        self.ra1t_dict = {
            'mslp': {
                'accumulate': False,
                'filename': 'umnsaa_pverb',
                'stash_name': 'air_pressure_at_sea_level',
                'stash_section': 16,
                'stash_item': 222
            },
            'air_temperature': {
                'accumulate': False,
                'filename': 'umnsaa_pvera',
                'stash_name': 'air_temperature',
                'stash_section': 3,
                'stash_item': 236
            },
            'cloud_fraction': {
                'accumulate': False,
                'filename': 'umnsaa_pverb',
                'stash_name': 'cloud_area_fraction_assuming_maximum_random_overlap',
                'stash_section': 9,
                'stash_item': 217
            },
            'precipitation': {
                'accumulate': True,
                'filename': 'umnsaa_pverb',
                'stash_name': 'stratiform_rainfall_rate',
                'stash_section': 4,
                'stash_item': 203
            },
            'relative_humidity': {
                'accumulate': False,
                'filename': 'umnsaa_pverc',
                'stash_name': 'relative_humidity',
                'stash_section': 16,
                'stash_item': 204
            },
            'wet_bulb_potential_temp': {
                'accumulate': False,
                'filename': 'umnsaa_pverc',
                'stash_name': 'wet_bulb_potential_temperature',
                'stash_section': 16,
                'stash_item': 205
            },
            'x_wind': {
                'accumulate': False,
                'filename': 'umnsaa_pvera',
                'stash_name': 'x_wind',
                'stash_section': 3,
                'stash_item': 225,
            },
            'x_winds_upper': {
                'accumulate': False,
                'filename': 'umnsaa_pverc024',
                'stash_name': 'x_wind',
                'stash_section': 15,
                'stash_item': 201
            },
            'y_wind': {
                'accumulate': False,
                'filename': 'umnsaa_pvera',
                'stash_name': 'y_wind',
                'stash_section': 3,
                'stash_item': 226,
            },
            'y_winds_upper': {
                'accumulate': False,
                'filename': 'umnsaa_pverc',
                'stash_name': 'y_wind',
                'stash_section': 15,
                'stash_item': 202
            }
        }
        self.ga6_dict = dict(self.ra1t_dict)
        self.ga6_dict['precipitation'] = {
            'accumulate': True,
            'filename': 'umnsaa_pverb',
            'stash_name': 'precipitation_flux',
            'stash_section': 5,
            'stash_item': 216
        }

    def test_ga6_stash_codes(self):
        result = forest.stash_codes("ga6")
        expect = self.ga6_dict
        self.assert_dict_equal(expect, result)

    def test_ra1t_stash_codes(self):
        result = forest.stash_codes("ra1t")
        self.assert_dict_equal(self.ra1t_dict, result)

    def test_ra1t_stash_section_given_variable(self):
        result = forest.stash_section('mslp', convention='ra1t')
        self.assertEqual(self.ra1t_dict['mslp']['stash_section'], result)

    def test_ra1t_stash_item_given_variable(self):
        result = forest.stash_item('mslp', convention='ra1t')
        self.assertEqual(self.ra1t_dict['mslp']['stash_item'], result)

    def test_ra1t_stash_name_given_variable(self):
        result = forest.stash_name('mslp', convention='ra1t')
        self.assertEqual(self.ra1t_dict['mslp']['stash_name'], result)

    def assert_dict_equal(self, expect, result):
        self.maxDiff = None
        self.assertEqual(expect.keys(), result.keys())
        for key in expect.keys():
            try:
                self.assertEqual(expect[key], result[key])
            except AssertionError as err:
                msg = "Value mismatch '{}': {}".format(key, err.args[0])
                err.args = (msg,)
                raise err
