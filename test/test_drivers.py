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
    dataset = driver.Dataset(label, pattern="*.json")
    loader = dataset.loader()
    assert isinstance(loader, rdt.Loader)
    assert isinstance(loader.locator, rdt.Locator)


def test_eida50_loader():
    driver = forest.drivers.by_name("eida50")
    dataset = driver.Dataset("Label", pattern="*.nc")
    loader = dataset.loader()
    assert isinstance(loader, eida50.Loader)
    assert isinstance(loader.locator, eida50.Locator)
