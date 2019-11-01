import os
import forest.drivers
from forest.drivers import (
        earth_networks,
        eida50,
        rdt)


def test_by_name_earth_networks():
    driver = forest.drivers.by_name("earth_networks")
    assert isinstance(driver.Dataset("label", pattern="engl*.txt"),
            earth_networks.Dataset)


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


def test_rdt_map_view():
    label = "RDT"
    dataset = rdt.Dataset(label, pattern="*.json")
    view = dataset.map_view(loader=None, color_mapper=None)
    assert isinstance(view, rdt.View)


def test_eida50_loader():
    driver = forest.drivers.by_name("eida50")
    dataset = driver.Dataset("Label", pattern="*.nc")
    loader = dataset.loader()
    assert isinstance(loader, eida50.Loader)
    assert isinstance(loader.locator, eida50.Locator)


def test_earth_networks_loader():
    driver = forest.drivers.by_name("earth_networks")
    dataset = driver.Dataset("Label", pattern="empty.txt")
    with open("empty.txt", "w"):
        pass
    loader = dataset.loader()
    os.remove("empty.txt")
    assert isinstance(loader, earth_networks.Loader)


def test_earth_networks_map_view():
    path = "empty.txt"
    dataset = earth_networks.Dataset("label", pattern=path)
    with open(path, "w"):
        pass
    view = dataset.map_view(dataset.loader(), color_mapper=None)
    os.remove(path)
    assert isinstance(view, earth_networks.View)
