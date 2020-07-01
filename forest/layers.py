"""
Layers
------

Users can select combinations of visualisations displayed across
multiple figures. Components for organising view layers associated
with each figure can be found here.

"""
import copy
import bokeh.models
import bokeh.layouts
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable, List
import forest.mark
import forest.state
import forest.actions
from forest import rx
from forest.redux import Action, State, Store
from forest.observe import Observable
from forest import colors
import forest.drivers
import forest.mark


SAVE_LAYER = "LAYERS_SAVE_LAYER"
ON_ADD = "LAYERS_ON_ADD"
ON_EDIT = "LAYERS_ON_EDIT"
ON_CLOSE = "LAYERS_ON_CLOSE"
ON_SAVE = "LAYERS_ON_SAVE"
ON_BUTTON_GROUP = "LAYERS_ON_BUTTON_GROUP"
SET_FIGURES = "LAYERS_SET_FIGURES"
SET_ACTIVE = "LAYERS_SET_ACTIVE"


def set_figures(n: int) -> Action:
    return {"kind": SET_FIGURES, "payload": n}


def save_layer(index, settings) -> Action:
    """Action to save layer settings"""
    return {"kind": SAVE_LAYER, "payload": {"index": index, "settings": settings}}


def on_button_group(row_index: int, active: List[int]) -> Action:
    return {
        "kind": ON_BUTTON_GROUP,
        "payload": {"row_index": row_index, "active": active}
    }


def set_active(row_index: int, active: List[int]) -> Action:
    return {
        "kind": SET_ACTIVE,
        "payload": {"row_index": row_index, "active": active}
    }


def on_add() -> Action:
    return {"kind": ON_ADD}


def on_edit(row_index: int) -> Action:
    return {"kind": ON_EDIT, "payload": row_index}


def on_close(row_index: int) -> Action:
    return {"kind": ON_CLOSE, "payload": row_index}


def on_save(settings: dict) -> Action:
    return {"kind": ON_SAVE, "payload": settings}


def middleware(store: Store, action: Action) -> Iterable[Action]:
    """Action generator given current state and action"""
    kind = action["kind"]
    if kind == ON_BUTTON_GROUP:
        payload = action["payload"]
        yield set_active(payload["row_index"], payload["active"])
    elif kind == ON_SAVE:
        if get_mode(store.state) == "edit":
            index = edit_index(store.state)
        else:
            index = next_index(store.state)
        yield save_layer(index, action["payload"])
    else:
        yield action


def get_mode(state):
    """Parse state into either add/edit"""
    node = state
    for key in ("layers", "mode"):
        node = node.get(key, {})
    return node.get("state", "add")


def edit_index(state):
    """Parse state into index"""
    node = state
    for key in ("layers", "mode"):
        node = node.get(key, {})
    return node.get("index", 0)


def next_index(state):
    """Parse state into index"""
    node = state
    for key in ("layers", "index"):
        node = node.get(key, {})
    indices = [key for key in node.keys()]
    if len(indices) == 0:
        return 0
    else:
        return max(indices) + 1


def reducer(state: State, action: Action) -> State:
    """Combine state and action to produce new state"""
    state = copy.deepcopy(state)
    if isinstance(state, dict):
        state = forest.state.State.from_dict(state)
    if isinstance(action, dict):
        try:
            action = forest.actions.Action.from_dict(action)
        except TypeError:
            return state.to_dict()

    if action.kind == SET_FIGURES:
        state.layers.figures = action.payload

    elif action.kind == ON_ADD:
        state.layers.mode.state = "add"

    elif action.kind == ON_CLOSE:
        row_index = action.payload
        try:
            layer_index = sorted(state.layers.index.keys())[row_index]
            del state.layers.index[layer_index]
        except IndexError:
            pass

    elif action.kind == ON_EDIT:
        row_index = action.payload
        layer_index = sorted(state.layers.index.keys())[row_index]
        state.layers.mode.state = "edit"
        state.layers.mode.index = layer_index

    elif action.kind == SAVE_LAYER:
        # NOTE: Layer index is stored in payload
        layer_index = action.payload["index"]
        settings = action.payload["settings"]
        if layer_index in state.layers.index:
            state.layers.index[layer_index].update(settings)
        else:
            state.layers.index[layer_index] = settings

    elif action.kind == SET_ACTIVE:
        active = action.payload["active"]
        row_index = action.payload["row_index"]
        row_to_layer = sorted(state.layers.index.keys())
        try:
            layer_index = row_to_layer[row_index]
            state.layers.index[layer_index]["active"] = active
        except IndexError:
            pass

    return state.to_dict()


def _connect(view, store):
    stream = (rx.Stream()
                .listen_to(store)
                .map(view.to_props)
                .filter(lambda x: x is not None)
                .distinct())
    stream.map(lambda props: view.render(*props))


@forest.mark.component
class FigureUI(Observable):
    """Controls how many figures are currently displayed"""
    def __init__(self, max_figures=3):
        self.max_figures = max_figures
        self.labels = [
            "Single figure",
            "Side by side",
            "3 way comparison"][:self.max_figures]
        self.select = bokeh.models.Select(
            options=self.labels,
            value=self.labels[0],
            width=350,
        )
        self.select.on_change("value", self.on_change)
        self.layout = bokeh.layouts.column(
            self.select,
        )
        super().__init__()

    def connect(self, store):
        self.add_subscriber(store.dispatch)
        store.add_subscriber(self.render)

    def render(self, state):
        if isinstance(state, dict):
            state = forest.state.State.from_dict(state)
        i = state.layers.figures - 1
        self.select.value = self.labels[i]

    def on_change(self, attr, old, new):
        """Emit action to set number of figures in state"""
        n = self.labels.index(new) + 1  # Select 0-indexed
        self.notify(set_figures(n))


class FigureRow:
    """Component to toggle number of displayed figures"""
    def __init__(self, figures):
        self.figures = figures
        self.layout = bokeh.layouts.row(*figures,
                sizing_mode="stretch_both",
                name="figures")
        self.layout.children = [self.figures[0]]  # Trick to keep correct sizing modes

    def connect(self, store):
        """Connect to the Store"""
        stream = (rx.Stream()
                    .listen_to(store)
                    .map(self.to_props)
                    .filter(lambda x: x is not None)
                    .distinct())
        stream.map(lambda props: self.render(*props))

    def to_props(self, state: State):
        """Select number of figures from state"""
        layers = state.get("layers", {})
        try:
            return (layers["figures"],)
        except KeyError:
            pass

    def render(self, n: int):
        """Assign figures to row"""
        if int(n) == 1:
            self.layout.children = [
                    self.figures[0]]
        elif int(n) == 2:
            self.layout.children = [
                    self.figures[0],
                    self.figures[1]]
        elif int(n) == 3:
            self.layout.children = [
                    self.figures[0],
                    self.figures[1],
                    self.figures[2]]


class OpacitySlider:
    def __init__(self):
        # Image opacity user interface (client-side)
        self.slider = bokeh.models.Slider(
            start=0,
            end=1,
            step=0.1,
            value=1.0,
            title="Image opacity",
            show_value=False)
        self.layout = bokeh.layouts.row(self.slider)

    def add_renderers(self, renderers):
        renderers = [r for r in renderers if self.is_image(r)]
        if len(renderers) == 0:
            return

        # Set initial alpha to slider value
        for renderer in renderers:
            renderer.glyph.global_alpha = self.slider.value

        # Pass server-side renderers to client-side callback
        custom_js = bokeh.models.CustomJS(
                args=dict(renderers=renderers),
                code="""
                renderers.forEach(function (r) {
                    r.glyph.global_alpha = cb_obj.value
                })
                """)
        self.slider.js_on_change("value", custom_js)

    @staticmethod
    def is_image(renderer):
        return isinstance(getattr(renderer, 'glyph', None), bokeh.models.Image)


@forest.mark.component
class LayersUI(Observable):
    """Collection of user interface components to manage layers"""
    def __init__(self):
        self.defaults = {
            "label": "Model/observation",
            "flags": [False, False, False],
            "figure": {
                1: ["Show"],
                2: ["L", "R"],
                3: ["L", "C", "R"]
            }
        }
        self.button_groups = []
        self.selects = []
        self.buttons = {
            "edit": [],
            "close": [],
            "add": bokeh.models.Button(label="New layer", width=110),
        }
        custom_js = bokeh.models.CustomJS(code="openModal()")
        self.buttons["add"].js_on_click(custom_js)
        self.buttons["add"].on_click(self.on_add)
        self.columns = {
            "rows": bokeh.layouts.column(),
            "buttons": bokeh.layouts.column(
                bokeh.layouts.row(self.buttons["add"])
            )
        }
        self.layout = bokeh.layouts.column(
            self.columns["rows"],
            self.columns["buttons"]
        )
        self._labels = ["Show"]
        super().__init__()

    def connect(self, store):
        """Connect component to store"""
        _connect(self, store)
        return self

    def to_props(self, state) -> tuple:
        """Select data from state that satisfies self.render(*props)"""
        layers = state.get("layers", {})
        return (
            self.parse_layers(state),
            layers.get("figures", None),
        )

    def parse_layers(self, state):
        if isinstance(state, dict):
            state = forest.state.State.from_dict(state)
        print(state.layers.index)
        return [value for _, value in sorted(state.layers.index.items())]

    def render(self, layers, figure_index):
        """Display latest application state in user interface

        :param n: integer representing number of rows
        """
        # Match rows to number of labels
        n = len(layers)
        nrows = len(self.columns["rows"].children) # - 1
        if n > nrows:
            # for label in labels[nrows:]:
            for _ in range(n - nrows):
                self.add_row()
        if n < nrows:
            for _ in range(nrows - n):
                self.remove_row()

        # Set button group labels
        if figure_index is not None:
            self.labels = self.defaults["figure"][figure_index]

        # Set options in select menus
        labels = [layer["label"] for layer in layers
                  if "label" in layer]
        options = list(sorted(labels))
        for select in self.selects:
            select.options = options

        # Set value for each select
        for i, layer in enumerate(layers):
            if "label" in layer:
                self.selects[i].value = layer["label"]
            if "active" in layer:
                self.button_groups[i].active = layer["active"]

    def on_add(self):
        """Event-handler when Add button is clicked"""
        self.notify(on_add())

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels
        for g in self.button_groups:
            g.labels = labels

    def add_row(self):
        """Add a bokeh.layouts.row with a dropdown and checkboxbuttongroup"""
        row_index = len(self.columns["rows"].children)

        widths = {
            "dropdown": 150,
            "button": 50,
            "group": 50,
            "row": 350
        }

        # Select
        select = bokeh.models.Select(width=widths["dropdown"], disabled=True)
        self.selects.append(select)

        # Edit button
        edit_button = bokeh.models.Button(label="Edit", width=widths["button"])
        custom_js = bokeh.models.CustomJS(code="openModal()")
        edit_button.js_on_click(custom_js)
        edit_button.on_click(self.on_edit(row_index))
        self.buttons["edit"].append(edit_button)

        # Close button
        close_button = bokeh.models.Button(label=u"\u274C", width=widths["button"])
        close_button.on_click(self.on_close(row_index))
        self.buttons["close"].append(close_button)

        # Button group
        button_group = bokeh.models.CheckboxButtonGroup(
                default_size=widths["group"],
                max_width=widths["group"],
                labels=self.labels,
                width=widths["group"])
        button_group.on_change("active", self.on_button_group(row_index))
        self.button_groups.append(button_group)

        # Row
        row = bokeh.layouts.row(edit_button,
                                close_button,
                                select,
                                button_group,
                                width=widths["row"])
        self.columns["rows"].children.append(row)

    def remove_row(self):
        """Remove a row from user interface"""
        if len(self.columns["rows"].children) > 0:
            self.selects.pop()
            self.button_groups.pop()
            self.buttons["edit"].pop()
            self.columns["rows"].children.pop()

    def on_edit(self, row_index: int):
        def _callback():
            self.notify(on_edit(row_index))
        return _callback

    def on_close(self, row_index: int):
        def _callback():
            self.notify(on_close(row_index))
        return _callback

    def on_button_group(self, row_index: int):
        """Translate event into Action"""
        def _callback(attr, old, new):
            # Note: bokeh.core.PropertyList can not be deep copied
            #       it RuntimeErrors, cast as list instead
            active = list(new)
            self.notify(on_button_group(row_index, active))
        return _callback


@dataclass
class LayerSpec:
    label: str = ""
    dataset: str = ""
    variable: str = ""
    active: List[int] = field(default_factory=list)
    color_spec: colors.ColorSpec = colors.ColorSpec()
    colorbar: dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.color_spec, dict):
            self.color_spec = colors.ColorSpec(**self.color_spec)


class Gallery:
    """Orchestration layer for MapViews"""
    def __init__(self, pools):
        self.pools = pools
        self.lock = False

    @classmethod
    def from_datasets(cls, datasets, factory_class):
        """Convenient constructor"""
        pools = {}
        for label, dataset in datasets.items():
            if hasattr(dataset, "map_view"):
                pools[label] = Pool(factory_class(dataset))
        return cls(pools)

    def connect(self, store):
        store.add_subscriber(self.render)

    def render(self, state):
        # Note: lock used to ignore state changes while building layers
        #       e.g. source limit events
        if not self.lock:
            self.lock = True
            self.render_specs(self.parse_specs(state), state)
            self.lock = False

    def parse_specs(self, state):
        # Parse application state
        node = state
        for key in ("layers", "index"):
            node = node.get(key, {})
        return [LayerSpec(**kwargs) for _, kwargs in sorted(node.items())]

    def render_specs(self, specs, state):
        # Dynamically build/use layers from object pools
        used_layers = defaultdict(list)
        for spec in specs:
            if spec.dataset == "":
                continue

            key = spec.dataset
            layer = self.pools[key].acquire()

            # Unmute layer
            layer.unmute()

            # Update figure visibility
            layer.active = spec.active

            # Layer-specific state
            layer_state = {}
            layer_state.update(state)
            if spec.variable != "":
                layer_state.update(variable=spec.variable,
                                   colorbar=spec.colorbar)
            layer.render(layer_state)

            used_layers[key].append(layer)

        # Mute unused layers
        for pool in self.pools.values():
            pool.map(lambda layer: layer.mute())

        # Return used layers to pool(s)
        for key, layers in used_layers.items():
            pool = self.pools[key]
            for layer in layers:
                pool.release(layer)

class Layer:
    """Facade to ease API"""
    def __init__(self, map_view, visible, source_limits):
        self.map_view = map_view
        self.image_sources = getattr(self.map_view, "image_sources", [])
        self.visible = visible
        self.source_limits = source_limits
        if self.source_limits is not None:
            for source in self.image_sources:
                self.source_limits.add_source(source)

    def render(self, state):
        self.map_view.render(state)

    def mute(self):
        self.active = []
        if self.source_limits is not None:
            for source in self.image_sources:
                self.source_limits.remove_source(source)

    def unmute(self):
        if self.source_limits is not None:
            for source in self.image_sources:
                self.source_limits.add_source(source)

    @property
    def active(self):
        return self.visible.active

    @active.setter
    def active(self, value):
        self.visible.active = value


def factory(*args):
    """Curry Factory constructor to accept a single argument"""
    def wrapper(dataset):
        return Factory(dataset, *args)
    return wrapper


class Factory:
    """Reusable layers

    Admittedly, there is a lot of coupling here that could be revised
    in future releases
    """
    def __init__(self, dataset,
                 color_mapper,
                 figures,
                 source_limits,
                 opacity_slider):
        self._calls = 0
        self.dataset = dataset
        self.color_mapper = color_mapper
        self.figures = figures
        self.source_limits = source_limits
        self.opacity_slider = opacity_slider

    def __call__(self):
        """Complex construction"""
        self._calls += 1
        print("Factory.__call__: {}".format(self._calls))
        try:
            map_view = self.dataset.map_view(self.color_mapper)
        except TypeError:
            map_view = self.dataset.map_view()
        visible = Visible.from_map_view(map_view, self.figures)
        if self.opacity_slider is not None:
            self.opacity_slider.add_renderers(visible.renderers)
        return Layer(map_view, visible, self.source_limits)


class Visible:
    """Wrapper to make MapView layers visible/invisible"""
    def __init__(self, renderers):
        self._active = []
        self.renderers = renderers
        for renderer in self.renderers:
            renderer.visible = False
            renderer.level = "underlay"

    @classmethod
    def from_map_view(cls, map_view, figures):
        """Construct from map_view and figures"""
        renderers = [map_view.add_figure(figure) for figure in figures]
        return cls(renderers)

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, indices):
        for i in range(len(self.renderers)):
            self.renderers[i].visible = i in indices
        self._active = indices


class Pool:
    """Manage reusable objects

    :param factory: function to create new objects
    """
    def __init__(self, factory):
        self.factory = factory
        self.reusables = deque()

    def map(self, func):
        """Apply function to objects inside pool"""
        for reusable in self.reusables:
            func(reusable)

    def acquire(self):
        """Select or create an object"""
        if len(self.reusables) == 0:
            self.reusables.appendleft(self.factory())
        return self.reusables.pop()

    def release(self, reusable):
        """Return object to Pool"""
        self.reusables.appendleft(reusable)
