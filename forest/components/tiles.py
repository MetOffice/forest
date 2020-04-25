"""Select web map tiling services to show map"""
import copy
import bokeh.models
from forest.observe import Observable
from forest.redux import Action, State


# Labels to identify tile servers
OPEN_STREET_MAP = "Open street map"
STAMEN_TERRAIN = "Stamen terrain"
STAMEN_WATERCOLOR = "Stamen watercolor"
STAMEN_TONER = "Stamen toner"
STAMEN_TONER_LITE = "Stamen toner lite"
WIKIMEDIA = "Wikimedia"

URLS = {
    WIKIMEDIA: "https://maps.wikimedia.org/osm-intl/{Z}/{X}/{Y}.png",
    OPEN_STREET_MAP: "http://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png",
    STAMEN_TERRAIN: "http://tile.stamen.com/terrain-background/{Z}/{X}/{Y}.png",
    STAMEN_WATERCOLOR: "http://tile.stamen.com/watercolor/{Z}/{X}/{Y}.jpg",
    STAMEN_TONER: "http://tile.stamen.com/toner-background/{Z}/{X}/{Y}.png",
    STAMEN_TONER_LITE: "http://tile.stamen.com/toner-lite/{Z}/{X}/{Y}.png"
}


def background_url(name):
    """Tile server URL for backgrounds"""
    return URLS[name]


def labels_url(name):
    """Tile server URL for labels"""
    default = "http://tile.stamen.com/toner-labels/{Z}/{X}/{Y}.png"
    return {
        STAMEN_TERRAIN: "http://tile.stamen.com/terrain-labels/{Z}/{X}/{Y}.png"
    }.get(name, default)


OPEN_STREET_MAP_ATTRIBUTION = """
© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors
"""


STAMEN_TONER_AND_TERRAIN_ATTRIBUTION = """
Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://creativecommons.org/licenses/by-sa/3.0">CC BY SA</a>.
"""


STAMEN_WATERCOLOR_ATTRIBUTION = """
Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://creativecommons.org/licenses/by-sa/3.0">CC BY SA</a>.
"""


WIKIMEDIA_ATTRIBUTION = """
Wikimedia Maps | © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors
"""


def attribution(name):
    """Tile server attribution"""
    if name in [
            STAMEN_TONER,
            STAMEN_TERRAIN,
            STAMEN_TONER_LITE]:
        return STAMEN_TONER_AND_TERRAIN_ATTRIBUTION
    elif name == STAMEN_WATERCOLOR:
        return STAMEN_WATERCOLOR_ATTRIBUTION
    elif name == WIKIMEDIA:
        return WIKIMEDIA_ATTRIBUTION
    elif name == OPEN_STREET_MAP:
        return OPEN_STREET_MAP_ATTRIBUTION
    else:
        return ""


SET_TILE = "TILES_SET_TILE"
SET_LABEL_VISIBLE = "TILES_SET_LABEL_VISIBLE"


def set_tile(provider: str) -> Action:
    """Action to choose tile provider"""
    return {"kind": SET_TILE, "payload": provider}


def set_label_visible(value: bool) -> Action:
    """Action to set tile labels on/off"""
    return {"kind": SET_LABEL_VISIBLE, "payload": value}


def reducer(state: State, action: Action) -> State:
    """Reducer to handle web map tiling settings"""
    state = copy.deepcopy(state)
    kind = action["kind"]
    if kind in [SET_TILE, SET_LABEL_VISIBLE]:
        tree = state.get("tile", {})
        if kind == SET_TILE:
            tree["name"] = action["payload"]
        elif kind == SET_LABEL_VISIBLE:
            tree["labels"] = action["payload"]
        state["tile"] = tree
    return state


class TilePicker(Observable):
    """Web map tile selector"""
    def __init__(self):
        self.tiles = {
            "underlay": bokeh.models.WMTSTileSource(
                url="https://maps.wikimedia.org/osm-intl/{Z}/{X}/{Y}.png",
                attribution=""),
            "labels": bokeh.models.WMTSTileSource(
                url="http://tile.stamen.com/toner-labels/{Z}/{X}/{Y}.png",
                attribution="")
        }
        self._renderers = []
        self.select = bokeh.models.Select(
            options=sorted(URLS.keys()),
            width=350)
        self.select.on_change("value", self.on_select)
        self.toggle = bokeh.models.Toggle(label="Show labels")
        self.toggle.on_click(self.on_toggle)
        self.layout = bokeh.layouts.column(
            self.select,
            self.toggle)
        super().__init__()

    def add_figure(self, figure):
        # Underlay tile
        renderer = figure.add_tile(self.tiles["underlay"])
        renderer.level = "underlay"
        # Overlay tile
        renderer = figure.add_tile(self.tiles["labels"])
        renderer.level = "overlay"
        renderer.alpha = 0
        self._renderers.append(renderer)

    def connect(self, store):
        """Connect component to store"""
        self.add_subscriber(store.dispatch)
        store.add_subscriber(self.render)
        return self

    def on_select(self, attr, old, new):
        """Send set tile action to store"""
        self.notify(set_tile(new))

    def on_toggle(self, value):
        """Send toggle tile labels action to store"""
        self.notify(set_label_visible(value))

    def render(self, state):
        """Represent state"""
        provider = state.get("tile", {}).get("name")
        if provider is None:
            return
        self.tiles["underlay"].url = URLS[provider]
        self.tiles["underlay"].attribution = attribution(provider)

        # Labels URL
        self.tiles["labels"].url = labels_url(provider)

        # Hide/show overlay labels
        visible = state.get("tile", {}).get("labels", False)
        if visible:
            alpha = 1
        else:
            alpha = 0
        for renderer in self._renderers:
            renderer.alpha = alpha

        # Select options
        if self.select.value != provider:
            self.select.value = provider

        # Toggle setting
        if self.toggle.active != visible:
            self.toggle.active = visible
