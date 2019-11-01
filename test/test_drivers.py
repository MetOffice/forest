import forest.drivers
from forest.drivers import (
        eida50,
        rdt)


def test_by_name_eida50():
    driver = forest.drivers.by_name("eida50")
    assert isinstance(driver.Dataset("label"), eida50.Dataset)


def test_by_name_rdt():
    driver = forest.drivers.by_name("rdt")
    assert isinstance(driver.Dataset("label"), rdt.Dataset)


def test_rdt_loader():
    driver = forest.drivers.by_name("rdt")
    label = "RDT"
    settings = {
        "pattern": "*.json"
    }
    dataset = driver.Dataset(label, settings)
    loader = dataset.loader()
    assert isinstance(loader, rdt.Loader)
    assert isinstance(loader.locator, rdt.Locator)
