from datetime import datetime
from unittest.mock import Mock, call, patch, sentinel
import unittest

import iris
import numpy as np

from forest import gridded_forecast


class Test_empty_image(unittest.TestCase):
    def test(self):
        result = gridded_forecast.empty_image()
        self.assertEqual(result.keys(), {'x', 'y', 'dw', 'dh', 'image', 'name',
                                         'units', 'valid', 'initial', 'length',
                                         'level'})
        for value in result.values():
            self.assertEqual(value, [])


class Test_to_datetime(unittest.TestCase):
    def test_datetime(self):
        dt = datetime.now()
        result = gridded_forecast._to_datetime(dt)
        self.assertEqual(result, dt)

    def test_str_with_space(self):
        result = gridded_forecast._to_datetime('2019-10-10 01:02:34')
        self.assertEqual(result, datetime(2019, 10, 10, 1, 2, 34))

    def test_str_iso8601(self):
        result = gridded_forecast._to_datetime('2019-10-10T01:02:34')
        self.assertEqual(result, datetime(2019, 10, 10, 1, 2, 34))

    def test_datetime64(self):
        dt = np.datetime64('2019-10-10T11:22:33')
        result = gridded_forecast._to_datetime(dt)
        self.assertEqual(result, datetime(2019, 10, 10, 11, 22, 33))

    def test_unsupported(self):
        with self.assertRaisesRegexp(Exception, 'Unknown value'):
            gridded_forecast._to_datetime(12)


@patch('forest.gridded_forecast._to_datetime')
class Test_coordinates(unittest.TestCase):
    def test_surface_and_times(self, to_datetime):
        valid = datetime(2019, 10, 10, 9)
        initial = datetime(2019, 10, 10, 3)
        to_datetime.side_effect = [valid, initial]

        result = gridded_forecast.coordinates(sentinel.valid, sentinel.initial,
                                              [], None)

        self.assertEqual(to_datetime.mock_calls, [call(sentinel.valid),
                                                  call(sentinel.initial)])
        self.assertEqual(result, {'valid': [valid], 'initial': [initial],
                                  'length': ['T+6'], 'level': ['Surface']})

    def test_surface_no_pressures(self, to_datetime):
        result = gridded_forecast.coordinates(None, None, [], 950)

        self.assertEqual(result['level'], ['Surface'])

    def test_surface_no_pressure(self, to_datetime):
        result = gridded_forecast.coordinates(None, None, [1000, 900], None)

        self.assertEqual(result['level'], ['Surface'])

    def test_pressure(self, to_datetime):
        result = gridded_forecast.coordinates(None, None, [1000, 900], 900)

        self.assertEqual(result['level'], ['900 hPa'])
    

class Test_is_valid_cube(unittest.TestCase):
    def setUp(self):
        lon = iris.coords.DimCoord(range(5), 'longitude')
        lat = iris.coords.DimCoord(range(4), 'latitude')
        time = iris.coords.DimCoord(range(3), 'time')
        other = iris.coords.DimCoord(range(2), long_name='other')
        frt = iris.coords.AuxCoord(range(1), 'forecast_reference_time')
        cube = iris.cube.Cube(np.empty((2, 3, 4, 5)), 'air_temperature',
                              dim_coords_and_dims=[(other, 0), (time, 1),
                                                   (lat, 2), (lon, 3)],
                              aux_coords_and_dims=[(frt, ())])
        self.cube = cube

    def test_ok(self):
        cube = self.cube[0]
        self.assertTrue(gridded_forecast._is_valid_cube(cube))

    def test_1d(self):
        cube = self.cube[0, 0, 0]
        self.assertFalse(gridded_forecast._is_valid_cube(cube))

    def test_4d(self):
        cube = self.cube
        self.assertFalse(gridded_forecast._is_valid_cube(cube))

    def test_2d_missing_time_coord(self):
        cube = self.cube[0, 0]
        cube.remove_coord(cube.coord('time'))
        self.assertFalse(gridded_forecast._is_valid_cube(cube))

    def test_missing_frt_coord(self):
        cube = self.cube[0]
        cube.remove_coord(cube.coord('forecast_reference_time'))
        self.assertFalse(gridded_forecast._is_valid_cube(cube))

    def test_missing_dim_coord(self):
        cube = self.cube[0]
        cube.remove_coord(cube.dim_coords[0])
        self.assertFalse(gridded_forecast._is_valid_cube(cube))

    def test_invalid_dim_coord(self):
        cube = self.cube[0]
        cube.dim_coords[2].rename('projection_x_coordinate')
        self.assertFalse(gridded_forecast._is_valid_cube(cube))

    def test_transposed(self):
        cube = self.cube[0]
        cube.transpose()
        self.assertFalse(gridded_forecast._is_valid_cube(cube))


class Test_load(unittest.TestCase):
    @patch('forest.gridded_forecast._is_valid_cube')
    @patch('iris.load')
    def test_all_unique(self, load, is_valid_cube):
        cube1 = Mock(**{'name.return_value': 'foo'})
        cube2 = Mock(**{'name.return_value': 'bar'})
        load.return_value = [cube1, cube2]
        is_valid_cube.return_value = True

        result = gridded_forecast._load(sentinel.pattern)

        load.assert_called_once_with(sentinel.pattern)
        self.assertEqual(is_valid_cube.mock_calls, [call(cube1), call(cube2)])
        self.assertEqual(result, {'foo': cube1, 'bar': cube2})

    @patch('forest.gridded_forecast._is_valid_cube')
    @patch('iris.load')
    def test_duplicate_name(self, load, is_valid_cube):
        cube1 = Mock(**{'name.return_value': 'foo'})
        cube2 = Mock(**{'name.return_value': 'foo'})
        load.return_value = [cube1, cube2]
        is_valid_cube.return_value = True

        result = gridded_forecast._load(sentinel.pattern)

        load.assert_called_once_with(sentinel.pattern)
        self.assertEqual(is_valid_cube.mock_calls, [call(cube1), call(cube2)])
        self.assertEqual(result, {'foo (1)': cube1, 'foo (2)': cube2})

    @patch('forest.gridded_forecast._is_valid_cube')
    @patch('iris.load')
    def test_none_valid(self, load, is_valid_cube):
        load.return_value = ['foo', 'bar']
        is_valid_cube.return_value = False

        with self.assertRaises(AssertionError):
            gridded_forecast._load(None)
