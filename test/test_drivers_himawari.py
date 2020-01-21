import iris
import forest
from forest.drivers import himawari


def test_load_driver():
    driver = forest.load_driver('himawari')
    assert hasattr(driver, 'Dataset')


def test_dataset():
    dataset = himawari.Dataset()
    loader = dataset.map_loader()


def test_dataset_navigator():
    dataset = himawari.Dataset()
    assert isinstance(dataset.navigator(), himawari.Navigator)


def test_load_a_cube():
    path = "/hpc/data/d03/frtb/simim_grib/umglobal-simvis_HIM8_20200120_gl12_T39.grib2"
    cubes = iris.load(path)
    cube = cubes[0]
    expect = [
        'latitude',
        'longitude',
        'forecast_period',
        'forecast_reference_time',
        'time'
    ]
    assert [coord.name() for coord in cube.coords()] == expect
