import pytest
import yaml
import bokeh.models
import forest
import forest.drivers
from forest import main, rdt


def test_rdt_loader_given_pattern():
    loader = forest.Loader.from_pattern("Label", "RDT*.json", "rdt")
    assert isinstance(loader, rdt.Loader)


def test_build_loader_given_files():
    settings = {"pattern": "file_20190101T0000Z.nc",
                "color_mapper": bokeh.models.ColorMapper()}
    dataset = forest.drivers.get_dataset("unified_model", settings)
    view = dataset.map_view()
    assert isinstance(view.loader, forest.drivers.unified_model.Loader)
    assert isinstance(view.loader.locator, forest.drivers.unified_model.Locator)


def test_build_loader_given_rdt_file_type():
    loader = forest.Loader.from_pattern(
            "Label", "*.json", "rdt")
    assert isinstance(loader, forest.rdt.Loader)
    assert isinstance(loader.locator, forest.rdt.Locator)
