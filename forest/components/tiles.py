"""Select web map tiling services to show map"""
import copy
import bokeh.models
from forest.observe import Observable


URLS = {
    "Wikimedia": "https://maps.wikimedia.org/osm-intl/{Z}/{X}/{Y}.png",
    "Open street maps": "http://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png",
    "OSM Terrain": "http://tile.stamen.com/terrain/{Z}/{X}/{Y}.jpg",
    "OSM Watercolor": "http://tile.stamen.com/watercolor/{Z}/{X}/{Y}.jpg",
    "Toner": "http://tile.stamen.com/toner/{Z}/{X}/{Y}.png",
    "Toner lite": "http://tile.stamen.com/toner-lite/{Z}/{X}/{Y}.png"
}
ATTRIBUTIONS = {
    "Wikimedia": "",
    "Open street maps": "",
    "OSM Terrain": "",
    "OSM Watercolor": """
Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://creativecommons.org/licenses/by-sa/3.0">CC BY SA</a>.
""",
    "Toner": "",
    "Toner lite": """
Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://creativecommons.org/licenses/by-sa/3.0">CC BY SA</a>.
"""
}


SET_TILE = "TILES_SET_TILE"
SET_LABELS = "TILES_SET_LABELS"


def set_tile(key):
    return {"kind": SET_TILE, "payload": key}


def set_labels(value):
    return {"kind": SET_LABELS, "payload": value}


def reducer(state, action):
    state = copy.deepcopy(state)
    tree = state.get("tile", {})
    if action["kind"] == SET_TILE:
        tree["name"] = action["payload"]
    elif action["kind"] == SET_LABELS:
        tree["labels"] = action["payload"]
    state["tile"] = tree
    return state


class TilePicker(Observable):
    """Web map tile selector"""
    def __init__(self, tile_source):
        self.tile_source = tile_source
        self.tiles = {
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
        renderer = figure.add_tile(self.tiles["labels"])
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
        self.notify(set_labels(value))

    def render(self, state):
        """Represent state"""
        key = state.get("tile", {}).get("name")
        if key is None:
            return
        self.tile_source.url = URLS[key]
        self.tile_source.attribution = ATTRIBUTIONS[key]

        # Hide/show overlay labels
        visible = state.get("tile", {}).get("labels", False)
        if visible:
            alpha = 1
        else:
            alpha = 0
        for renderer in self._renderers:
            renderer.alpha = alpha
