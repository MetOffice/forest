import pytest
from forest import drivers


@pytest.mark.parametrize(
    "driver_name", ["earth_networks", "ghrsstl4", "gridded_forecast"]
)
def test_singleton_dataset(driver_name):
    datasets = (
        drivers.get_dataset(driver_name),
        drivers.get_dataset(driver_name),
    )
    assert id(datasets[0]) == id(datasets[1])


@pytest.fixture
def fake_module(fake_driver_name):
    import sys
    from types import ModuleType

    module = ModuleType(fake_driver_name)

    class Dataset:
        """Fake Dataset"""

        def __init__(self, key):
            self.key = key

    module.Dataset = Dataset

    sys.modules[module.__name__] = module

    yield module

    del sys.modules[module.__name__]


@pytest.fixture
def fake_driver_name():
    return "a.b.c"


def test_get_dataset_given_injectable_driver(fake_module, fake_driver_name):
    actual = drivers.get_dataset(fake_driver_name, {"key": "value"})
    assert actual.key == "value"
