import pytest
from forest.components import tiles


@pytest.mark.parametrize("name,expect", [
    (tiles.OPEN_STREET_MAP, "http://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png"),
    (tiles.STAMEN_TERRAIN, "http://tile.stamen.com/terrain-background/{Z}/{X}/{Y}.jpg"),
    (tiles.STAMEN_WATERCOLOR, "http://tile.stamen.com/watercolor/{Z}/{X}/{Y}.jpg"),
    (tiles.STAMEN_TONER, "http://tile.stamen.com/toner-background/{Z}/{X}/{Y}.png"),
    (tiles.STAMEN_TONER_LITE, "http://tile.stamen.com/toner-lite/{Z}/{X}/{Y}.png"),
    (tiles.WIKIMEDIA, "https://maps.wikimedia.org/osm-intl/{Z}/{X}/{Y}.png"),
])
def test_background_url(name, expect):
    assert tiles.background_url(name) == expect


@pytest.mark.parametrize("name,expect", [
    (tiles.STAMEN_TERRAIN, "http://tile.stamen.com/terrain-labels/{Z}/{X}/{Y}.jpg"),
    (tiles.STAMEN_WATERCOLOR, "http://tile.stamen.com/toner-labels/{Z}/{X}/{Y}.png"),
    (tiles.STAMEN_TONER, "http://tile.stamen.com/toner-labels/{Z}/{X}/{Y}.png"),
    (tiles.STAMEN_TONER_LITE, "http://tile.stamen.com/toner-labels/{Z}/{X}/{Y}.png"),
])
def test_labels_url(name, expect):
    assert tiles.labels_url(name) == expect


@pytest.mark.parametrize("name,expect", [
    (tiles.STAMEN_TERRAIN, ""),
    (tiles.STAMEN_WATERCOLOR, ""),
])
def test_attribution(name, expect):
    assert tiles.attribution(name) == expect
