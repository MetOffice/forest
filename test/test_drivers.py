import pytest
from forest import drivers


@pytest.mark.parametrize("driver_name", ["earth_networks", "ghrsstl4", "gridded_forecast"])
def test_singleton_dataset(driver_name):
    datasets = (
        drivers.get_dataset(driver_name),
        drivers.get_dataset(driver_name))
    assert id(datasets[0]) == id(datasets[1])
