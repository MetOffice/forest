import pytest
import datetime as dt
import bokeh.models
import forest.drivers
import forest.drivers.gpm


def test_gpm_dataset():
    color_mapper = bokeh.models.ColorMapper()
    dataset = forest.drivers.get_dataset("gpm")
    navigator = dataset.navigator()
    map_view = dataset.map_view(color_mapper)
    map_view.render({})
    assert hasattr(map_view, "image_sources")


@pytest.mark.parametrize("paths,date,expect", [
    pytest.param([], dt.datetime.now(), [], id="Empty list"),
    pytest.param(
        ["20200101.nc", "20200102.nc"],
        dt.datetime(2020, 1, 1),
        ["20200101.nc"], id="Match a file")
])
def test_locator_find_paths(paths, date, expect):
    window_size = dt.timedelta(days=1)
    locator = forest.drivers.gpm.Locator()
    assert list(locator.find_paths(paths, date, window_size)) == expect
