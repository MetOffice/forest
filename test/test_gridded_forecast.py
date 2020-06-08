import pytest
import cftime
from datetime import datetime
from unittest.mock import Mock, call, patch, sentinel
import unittest

import iris
import numpy as np

from forest.drivers import gridded_forecast


class Test_empty_image(unittest.TestCase):
    def test(self):
        result = gridded_forecast.empty_image()
        self.assertEqual(result.keys(), {'x', 'y', 'dw', 'dh', 'image', 'name',
                                         'units', 'valid', 'initial', 'length',
                                         'level'})
        for value in result.values():
            self.assertEqual(value, [])


@patch('forest.drivers.gridded_forecast._to_datetime')
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
    @patch('forest.drivers.gridded_forecast._is_valid_cube')
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

    @patch('forest.drivers.gridded_forecast._is_valid_cube')
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

    @patch('forest.drivers.gridded_forecast._is_valid_cube')
    @patch('iris.load')
    def test_none_valid(self, load, is_valid_cube):
        load.return_value = ['foo', 'bar']
        is_valid_cube.return_value = False

        with self.assertRaises(AssertionError):
            gridded_forecast._load(None)


class Test_ImageLoader(unittest.TestCase):
    def test_init(self):
        result = gridded_forecast.ImageLoader(sentinel.label, sentinel.cubes)
        self.assertEqual(result._label, sentinel.label)
        self.assertEqual(result._cubes, sentinel.cubes)

    @patch('forest.drivers.gridded_forecast.empty_image')
    @patch('iris.Constraint')
    @patch('forest.drivers.gridded_forecast._to_datetime')
    def test_empty(self, to_datetime, constraint, empty_image):
        original_cube = Mock()
        original_cube.extract.return_value = None
        image_loader = gridded_forecast.ImageLoader(sentinel.label,
                                                    {'foo': original_cube})

        to_datetime.return_value = sentinel.valid_datetime
        constraint.return_value = sentinel.constraint
        empty_image.return_value = sentinel.empty_image

        result = image_loader.image(
            Mock(variable='foo', valid_time=sentinel.valid))

        to_datetime.assert_called_once_with(sentinel.valid)
        constraint.assert_called_once_with(time=sentinel.valid_datetime)
        original_cube.extract.assert_called_once_with(sentinel.constraint)
        self.assertEqual(result, sentinel.empty_image)

    @patch('forest.drivers.gridded_forecast.coordinates')
    @patch('forest.geo.stretch_image')
    @patch('iris.Constraint')
    @patch('forest.drivers.gridded_forecast._to_datetime')
    def test_image(self, to_datetime, constraint, stretch_image, coordinates):
        cube = Mock()
        cube.coord.side_effect = [Mock(points=sentinel.longitudes),
                                  Mock(points=sentinel.latitudes)]
        cube.units.__str__ = lambda self: 'my-units'
        original_cube = Mock()
        original_cube.extract.return_value = cube
        image_loader = gridded_forecast.ImageLoader('my-label',
                                                    {'foo': original_cube})

        to_datetime.return_value = sentinel.valid_datetime
        constraint.return_value = sentinel.constraint
        stretch_image.return_value = {'stretched_image': True}
        coordinates.return_value = {'coordinates': True}

        state = Mock(variable='foo',
                     valid_time=sentinel.valid,
                     initial_time=sentinel.initial,
                     pressures=sentinel.pressures,
                     pressure=sentinel.pressure)
        result = image_loader.image(state)

        self.assertEqual(cube.coord.mock_calls, [call('longitude'),
                                                 call('latitude')])
        stretch_image.assert_called_once_with(sentinel.longitudes,
                                              sentinel.latitudes, cube.data)
        coordinates.assert_called_once_with(sentinel.valid, sentinel.initial,
                                            sentinel.pressures,
                                            sentinel.pressure)
        self.assertEqual(result, {'stretched_image': True, 'coordinates': True,
                                  'name': ['my-label'], 'units': ['my-units']})


class Test_Navigator(unittest.TestCase):
    def test_init(self):
        result = gridded_forecast.Navigator(sentinel.cubes)
        self.assertEqual(result._cubes, sentinel.cubes)

    def test_variables(self):
        navigator = Mock(_cubes={'one': 1, 'two': 2, 'three': 3})
        result = gridded_forecast.Navigator.variables(navigator, None)
        self.assertEqual(list(sorted(result)), ['one', 'three', 'two'])

    def test_initial_times(self):
        cube1 = Mock()
        cube1.coord.return_value.cell().point = '2019-10-18'
        cube2 = Mock()
        cube2.coord.return_value.cell().point = '2019-10-19'
        cube3 = Mock()
        cube3.coord.return_value.cell().point = '2019-10-18'
        navigator = Mock()
        navigator._cubes.values.return_value = [cube1, cube2, cube3]

        result = gridded_forecast.Navigator.initial_times(navigator, None,
                                                          None)

        navigator._cubes.values.assert_called_once_with()
        cube1.coord.assert_called_once_with('forecast_reference_time')
        cube2.coord.assert_called_once_with('forecast_reference_time')
        cube3.coord.assert_called_once_with('forecast_reference_time')
        self.assertEqual(list(sorted(result)), ['2019-10-18', '2019-10-19'])

    def test_valid_times(self):
        cube1 = Mock()
        cube1.coord.return_value.cells.return_value = [Mock(point='p1'),
                                                       Mock(point='p2')]
        navigator = Mock()
        navigator._cubes = {'first': cube1, 'second': None}

        result = gridded_forecast.Navigator.valid_times(navigator, None,
                                                        'first', None)

        cube1.coord.assert_called_once_with('time')
        self.assertEqual(result, ['p1', 'p2'])

    def test_pressures(self):
        cube1 = Mock()
        cube1.coord.return_value.cells.return_value = [Mock(point='p1'),
                                                       Mock(point='p2')]
        navigator = Mock()
        navigator._cubes = {'first': cube1, 'second': None}
        result = gridded_forecast.Navigator.pressures(navigator, None,
                                                        'first', None)

        cube1.coord.assert_called_once_with('pressure')
        self.assertEqual(result, ['p1', 'p2'])

    def test_pressures_empty(self):
        cube1 = Mock()
        cube1.coord.side_effect = iris.exceptions.CoordinateNotFoundError
        navigator = Mock()
        navigator._cubes = {'first': cube1, 'second': None}

        result = gridded_forecast.Navigator.pressures(navigator, None,
                                                        'first', None)

        cube1.coord.assert_called_once_with('pressure')
        self.assertEqual(result, [])
