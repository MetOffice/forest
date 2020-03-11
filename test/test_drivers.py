from forest import drivers


def test_singleton_dataset():
    driver_name = "earth_networks"
    datasets = (
        drivers.get_dataset(driver_name),
        drivers.get_dataset(driver_name))
    assert id(datasets[0]) == id(datasets[1])
