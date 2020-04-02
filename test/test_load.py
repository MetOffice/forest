import pytest
import yaml
import bokeh.models
import forest.drivers
from forest.drivers import rdt


def test_build_loader_given_files():
    settings = {"pattern": "file_20190101T0000Z.nc"}
    color_mapper = bokeh.models.ColorMapper()
    dataset = forest.drivers.get_dataset("unified_model", settings)
    view = dataset.map_view(color_mapper)
    assert isinstance(view.loader, forest.drivers.unified_model.Loader)
    assert isinstance(view.loader.locator, forest.drivers.unified_model.Locator)


def test_build_loader_given_rdt_file_type():
    dataset = forest.drivers.get_dataset("rdt", {"label": "Label",
                                                 "pattern": "*.json"})
    map_view = dataset.map_view()
    assert isinstance(map_view.loader, rdt.Loader)
    assert isinstance(map_view.loader.locator, rdt.Locator)
