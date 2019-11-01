import forest.drivers
from forest.drivers import (
        eida50,
        rdt)


def test_by_name_eida50():
    driver = forest.drivers.by_name("eida50")
    assert isinstance(driver.Dataset(), eida50.Dataset)


def test_by_name_rdt():
    driver = forest.drivers.by_name("rdt")
    assert isinstance(driver.Dataset(), rdt.Dataset)
