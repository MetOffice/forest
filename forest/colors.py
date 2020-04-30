"""
Color palette
-------------

Helpers to choose color palette(s), limits etc.

UI components
~~~~~~~~~~~~~

The following components wire up the various bokeh
widgets and event handlers to actions and react to
changes in state. They are typically used in the
following manner.

>>> component = Component().connect(store)
>>> bokeh.layouts.column(component.layout)

.. autoclass:: ColorPalette
    :members:

.. autoclass:: UserLimits
    :members:

.. autoclass:: SourceLimits
    :members:

Most components are not interested in the full application
state. The :func:`connect` method and :func:`state_to_props`
are provided to only notify UI components when relevant state
updates.

.. autofunction:: connect

.. autofunction:: state_to_props


Reducer
~~~~~~~

A reducer combines the current state with an action
to produce a new state

.. autofunction:: reducer

Middleware
~~~~~~~~~~

Middleware pre-processes actions prior to the reducer

.. autofunction:: palettes

Helpers
~~~~~~~

Convenient functions to simplify color bar settings

.. autofunction:: defaults

.. autofunction:: palette_names

.. autofunction:: palette_numbers

Actions
~~~~~~~

Actions are small pieces of data used to communicate
with other parts of the system. Reducers and
middleware functions can interpret their contents
and either update state or generate new actions

.. autofunction:: set_colorbar

.. autofunction:: set_reverse

.. autofunction:: set_palette_name

.. autofunction:: set_palette_names

.. autofunction:: set_palette_number

.. autofunction:: set_palette_numbers

.. autofunction:: set_source_limits

.. autofunction:: set_user_high

.. autofunction:: set_user_low

.. autofunction:: set_invisible_min

.. autofunction:: set_invisible_max

"""
import copy
import bokeh.palettes
import bokeh.colors
import bokeh.layouts
import numpy as np
from forest.observe import Observable
from forest.rx import Stream
from forest.db.util import autolabel
from dataclasses import dataclass, asdict


def colorbar_figure(color_mapper, plot_width=500):
    # Dimensions
    padding = 5
    margin = 20
    colorbar_height = 20
    plot_height = colorbar_height + 30

    # Colorbar
    colorbar = bokeh.models.ColorBar(
        color_mapper=color_mapper,
        location=(0, 0),
        height=colorbar_height,
        width=int(plot_width - (margin + padding)),
        padding=padding,
        orientation="horizontal",
        major_tick_line_color="black",
        bar_line_color="black",
        background_fill_alpha=0.,
    )
    colorbar.title = ""

    # Figure
    figure = bokeh.plotting.figure(
        plot_height=plot_height,
        plot_width=plot_width,
        toolbar_location=None,
        min_border=0,
        background_fill_alpha=0,
        border_fill_alpha=0,
        outline_line_color=None,
    )
    figure.axis.visible = False
    figure.add_layout(colorbar, 'center')
    return figure



@dataclass
class ColorSpec:
    """Specifies color mapper settings"""
    name: str = "Greys"
    number: int = 256
    reverse: bool = False
    low: float = 0.
    low_visible: bool = True
    high: float = 1.
    high_visible: bool = True

    def __post_init__(self):
        if self.name == "Palettes":
            self.name = "Greys"
        try:
            self.number = int(self.number)
        except ValueError:
            self.number = 256
        self.low = float(self.low)
        self.high = float(self.high)

    @property
    def palette(self):
        if self.reverse:
            step = -1
        else:
            step = 1
        return bokeh.palettes.all_palettes[self.name][self.number][::step]

    @property
    def high_color(self):
        if self.high_visible:
            return None
        else:
            return bokeh.colors.RGB(0, 0, 0, a=0)

    @property
    def low_color(self):
        if self.low_visible:
            return None
        else:
            return bokeh.colors.RGB(0, 0, 0, a=0)

    def apply(self, color_mapper):
        """Helper to apply settings to color_mapper"""
        color_mapper.palette = self.palette
        color_mapper.low = self.low
        color_mapper.low_color = self.low_color
        color_mapper.high = self.high
        color_mapper.high_color = self.high_color


def parse_color_spec(props):
    kwargs = {}

    # Palette
    if "name" in props:
        kwargs["name"] = props["name"]
    if "number" in props:
        kwargs["number"] = props["number"]
    if "reverse" in props:
        kwargs["reverse"] = props["reverse"]
    if "invisible_min" in props:
        kwargs["low_visible"] = not props["invisible_min"]
    if "invisible_max" in props:
        kwargs["high_visible"] = not props["invisible_max"]

    # Limits
    origin = props.get("limits", {}).get("origin", "column_data_source")
    attrs = props.get("limits", {}).get(origin, {})
    if "low" in attrs:
        try:
            kwargs["low"] = float(attrs["low"])
        except ValueError:
            pass
    if "high" in attrs:
        try:
            kwargs["high"] = float(attrs["high"])
        except ValueError:
            pass
    return ColorSpec(**kwargs)


SET_INVISIBLE = "SET_INVISIBLE"
SET_PALETTE = "SET_PALETTE"
SET_LIMITS = "SET_LIMITS"
SET_LIMITS_ORIGIN = "SET_LIMITS_ORIGIN"


def set_colorbar(options):
    """Action to set multiple settings at once"""
    return {"kind": SET_PALETTE, "payload": options}


def set_reverse(flag):
    """Action to reverse color palette colors"""
    return {"kind": SET_PALETTE, "payload": {"reverse": flag}}


def set_palette_name(name):
    """Action to set color palette name"""
    return {"kind": SET_PALETTE, "payload": {"name": name}}


def set_palette_names(names):
    """Action to set all available palettes"""
    return {"kind": SET_PALETTE, "payload": {"names": names}}


def set_palette_number(number):
    """Action to set color palette size"""
    return {"kind": SET_PALETTE, "payload": {"number": number}}


def set_palette_numbers(numbers):
    """Action to set available levels for color palette"""
    return {"kind": SET_PALETTE, "payload": {"numbers": numbers}}


def set_source_limits(low, high):
    """Action to set colorbar limits from column data sources"""
    return {"kind": SET_LIMITS,
            "payload": {"low": low, "high": high},
            "meta": {"origin": "column_data_source"}}


def set_user_high(high):
    """Action to set user defined colorbar higher limit"""
    return {"kind": SET_LIMITS,
            "payload": {"high": high},
            "meta": {"origin": "user"}}


def set_user_low(low):
    """Action to set user defined colorbar lower limit"""
    return {"kind": SET_LIMITS,
            "payload": {"low": low},
            "meta": {"origin": "user"}}


def set_limits_origin(text):
    """Action to set limits origin, e.g. user/column_data_source"""
    return {"kind": SET_LIMITS_ORIGIN, "payload": text}


def set_invisible_min(flag):
    """Action to mask out data below colour bar limits"""
    return {"kind": SET_INVISIBLE, "payload": {"invisible_min": flag}}

def set_invisible_max(flag):
    """Action to mask out data below colour bar limits"""
    return {"kind": SET_INVISIBLE, "payload": {"invisible_max": flag}}


def reducer(state, action):
    """Reducer for colorbar actions

    Combines current state with an action to
    produce the next state

    :returns: new state
    :rtype: dict
    """
    state = copy.deepcopy(state)
    kind = action["kind"]
    if kind in [SET_PALETTE, SET_INVISIBLE]:
        state["colorbar"] = state.get("colorbar", {})
        state["colorbar"].update(action["payload"])
    return state


def limits_reducer(state, action):
    state = copy.deepcopy(state)

    if action["kind"] == SET_LIMITS_ORIGIN:
        # Build/traverse tree
        node = state
        keys = ("colorbar", "limits")
        for key in keys:
            node[key] = node.get(key, {})
            node = node[key]
        node.update({"origin": action["payload"]})

    elif meta_origin(action) in {"user", "column_data_source"}:
        # Build/traverse tree
        node = state
        keys = ("colorbar", "limits", meta_origin(action))
        for key in keys:
            node[key] = node.get(key, {})
            node = node[key]
        node.update(action["payload"])
        return state

    return state


def meta_origin(action):
    return action.get("meta", {}).get("origin", "")


def defaults():
    """Default color palette settings

    .. code-block:: python

        {
            "name": "Viridis",
            "names": palette_names(),
            "number": 256,
            "numbers": palette_numbers("Viridis"),
            "low": 0,
            "high": 1,
            "reverse": False,
            "invisible_min": False,
            "invisible_max": False,
        }

    .. note:: incomplete settings create unintuitive behaviour when restoring
              from a previously saved palette

    :returns: dict representing default colorbar
    """
    return {
        "name": "Viridis",
        "names": palette_names(),
        "number": 256,
        "numbers": palette_numbers("Viridis"),
        "low": 0,
        "high": 1,
        "reverse": False,
        "invisible_min": False,
        "invisible_max": False,
    }


def complete(settings):
    """Check current colorbar state is complete

    :returns: True if every colorbar setting is present
    """
    return all([key in settings for key in defaults().keys()])


def palette_names():
    """All palette names

    :returns: list of valid bokeh palette names
    """
    return list(sorted(bokeh.palettes.all_palettes.keys()))


def palettes(store, action):
    """Color palette middleware

    Encapsulates colorbar user interface logic. For example,
    if a user has chosen to fix their data limits, then
    set_limit actions generated by column data source changes
    are ignored

    .. note:: middleware is an action generator
    """
    kind = action["kind"]
    if kind == SET_PALETTE:
        payload = action["payload"]
        if "name" in payload:
            name = payload["name"]
            numbers = palette_numbers(name)
            yield set_palette_numbers(numbers)
            if "colorbar" in store.state:
                if "number" in store.state["colorbar"]:
                    number = store.state["colorbar"]["number"]
                    if number not in numbers:
                        yield set_palette_number(max(numbers))
        yield action
    elif kind == SET_LIMITS:
        yield action
    else:
        # While handling generic action augment with defaults if not already set
        yield action
        settings = store.state.get("colorbar", {})
        if not complete(settings):
            yield set_colorbar({**defaults(), **settings})


def middleware():
    previous = None
    seen = False
    def call(store, action):
        nonlocal previous, seen
        if not seen:
            seen = True
            previous = action
            yield action
        elif previous != action:
            previous = action
            yield action
    return call


def palette_numbers(name):
    """Helper to choose available color palette numbers

    :returns: list of valid bokeh palette numbers
    """
    return list(sorted(bokeh.palettes.all_palettes[name].keys()))


class SourceLimits(Observable):
    """Event stream listening to collection of ColumnDataSources

    Translates column data source on_change events into domain
    specific actions, e.g. :func:`set_source_limits`. Instead
    of connecting to a :class:`forest.redux.Store`, simply
    subscribe ``store.dispatch`` to action events.

    >>> source_limits = SourceLimits()
    >>> for source in sources:
    ...     source_limits.add_source(source)
    >>> source_limits.connect(store)

    .. note:: Unlike a typical component there is no ``layout`` property
              to attach to a bokeh document
    """
    def __init__(self):
        self.sources = []
        super().__init__()

    def connect(self, store):
        """Connect events to the Store"""
        self.add_subscriber(store.dispatch)
        return self

    def add_source(self, source):
        """Add ColumnDataSource to listened sources"""
        if source not in self.sources:
            source.on_change("data", self.on_change)
            self.sources.append(source)

    def remove_source(self, source):
        """Remove ColumnDataSource from listened sources"""
        # Remove on_change handler
        try:
            source.remove_on_change("data", self.on_change)
        except ValueError:
            pass

        # Remove source from limit calculation
        if source in self.sources:
            self.sources = [_source for _source in self.sources
                            if _source.id != source.id]
            low, high = self.limits(self.sources)
            self.notify(set_source_limits(low, high))

    def on_change(self, attr, old, new):
        """Generate action from bokeh event"""
        low, high = self.limits(self.sources)
        self.notify(set_source_limits(low, high))

    def limits(self, sources):
        """Calculate limits from underlying sources"""
        images = []
        for source in sources:
            if len(source.data["image"]) == 0:
                continue
            images.append(source.data["image"][0])
        if len(images) > 0:
            low = np.min([np.min(x) for x in images])
            high = np.max([np.max(x) for x in images])
            return low, high
        else:
            return 0, 1


class UserLimits(Observable):
    """User controlled color mapper limits"""
    def __init__(self):
        self.inputs = {
            "low": bokeh.models.TextInput(title="User min:",
                                          placeholder="Enter a number"),
            "high": bokeh.models.TextInput(title="User max:",
                                           placeholder="Enter a number"),
            "source_low": bokeh.models.TextInput(title="Data min:",
                                                 disabled=True),
            "source_high": bokeh.models.TextInput(title="Data max:",
                                                  disabled=True)
        }
        self.inputs["low"].on_change("value", self.on_input_low)
        self.inputs["high"].on_change("value", self.on_input_high)

        # RadioGroup for user/data limits
        self.radio_group = bokeh.models.RadioGroup(
            labels=["Use data limits", "Use user limits"],
            active=0,
            inline=True)
        self.radio_group.on_change("active", self.on_origin)

        self.checkboxes = {}

        # Checkbox transparency lower threshold
        self.checkboxes["invisible_min"] = bokeh.models.CheckboxGroup(
            labels=["Set data below Min to transparent"],
            active=[])
        self.checkboxes["invisible_min"].on_change("active", self.on_invisible_min)

        # Checkbox transparency upper threshold
        self.checkboxes["invisible_max"] = bokeh.models.CheckboxGroup(
            labels=["Set data above Max to transparent"],
            active=[])
        self.checkboxes["invisible_max"].on_change("active", self.on_invisible_max)

        widths = {
            "row": 310
        }
        self.layout = bokeh.layouts.column(
            bokeh.layouts.row(
                self.inputs["low"],
                self.inputs["high"],
                width=widths["row"]),
            bokeh.layouts.row(
                self.inputs["source_low"],
                self.inputs["source_high"],
                width=widths["row"]),
            bokeh.layouts.row(
                self.radio_group,
                width=widths["row"]),
            self.checkboxes["invisible_min"],
            self.checkboxes["invisible_max"],
        )
        super().__init__()

    def connect(self, store):
        """Connect component to Store

        Convert state stream to properties used
        by render method.

        :param store: instance to dispatch actions and listen to state changes
        :type store: :class:`forest.redux.Store`
        """
        connect(self, store)
        return self

    def on_input_low(self, attr, old, new):
        """Event-handler to set user low"""
        self.notify(set_user_low(new))

    def on_input_high(self, attr, old, new):
        """Event-handler to set user high"""
        self.notify(set_user_high(new))

    def on_invisible_min(self, attr, old, new):
        """Event-handler when invisible_min toggle is changed"""
        self.notify(set_invisible_min(len(new) == 1))

    def on_invisible_max(self, attr, old, new):
        """Event-handler when invisible_max toggle is changed"""
        self.notify(set_invisible_max(len(new) == 1))

    def on_origin(self, attr, old, new):
        origin = {1: "user"}.get(new, "column_data_source")
        self.notify(set_limits_origin(origin))

    def props(self):
        """Helper to get current state of widgets"""
        _props = {
            "limits": {
                "origin": {
                    0: "column_data_source",
                    1: "user"
                }[self.radio_group.active],
                "user": {},
                "column_data_source": {},
            }
        }

        # User inputs
        if self.inputs["high"].value is not None:
            _props["limits"]["user"]["high"] = self.inputs["high"].value
        if self.inputs["low"].value is not None:
            _props["limits"]["user"]["low"] = self.inputs["low"].value

        # ColumnDataSource inputs
        if self.inputs["source_high"].value is not None:
            _props["limits"]["column_data_source"]["high"] = self.inputs["source_high"].value
        if self.inputs["source_low"].value is not None:
            _props["limits"]["column_data_source"]["low"] = self.inputs["source_low"].value

        # Invisible min/max
        for key in ("invisible_min", "invisible_max"):
            _props[key] = len(self.checkboxes[key].active) == 1
        return _props

    def render(self, props):
        """Update user-defined limits inputs"""
        for key in ["invisible_min", "invisible_max"]:
            if props.get(key, False):
                self.checkboxes[key].active = [0]
            else:
                self.checkboxes[key].active = []

        # User limits
        origin = "user"
        attrs = props.get("limits", {}).get(origin, {})
        if "high" in attrs:
            self.inputs["high"].value = str(attrs["high"])
        if "low" in attrs:
            self.inputs["low"].value = str(attrs["low"])

        # ColumnDataSource limits
        origin = "column_data_source"
        attrs = props.get("limits", {}).get(origin, {})
        if "high" in attrs:
            self.inputs["source_high"].value = str(attrs["high"])
        if "low" in attrs:
            self.inputs["source_low"].value = str(attrs["low"])

        # Sync radio group
        origin = props.get("limits", {}).get("origin", "column_data_source")
        if origin == "column_data_source":
            self.radio_group.active = 0
        else:
            self.radio_group.active = 1


def state_to_props(state):
    """Map state to props relevant to component

    :param state: dict representing full application state
    :returns: ``state["colorbar"]`` or ``None``
    """
    return state.get("colorbar", None)


def connect(view, store):
    """Connect component to Store

    UI components connected to a Store
    only need to be notified when a change occurs that
    is relevant to them, all other state updates can be
    safely ignored.

    To implement component specific updates this helper method
    listens to store dispatch events, converts them
    to a stream of states, maps the states to
    props and filters out duplicates.
    """
    view.add_subscriber(store.dispatch)
    one_way_connect(view, store)


def one_way_connect(view, store):
    stream = (Stream()
                .listen_to(store)
                .map(state_to_props)
                .filter(lambda x: x is not None)
                .distinct())
    stream.map(lambda props: view.render(props))


class ColorMapperView:
    def __init__(self, color_mapper):
        self.color_mapper = color_mapper

    def connect(self, store):
        """Connect component to Store"""
        one_way_connect(self, store)
        return self

    def render(self, props):
        if isinstance(props, ColorSpec):
            spec = props
        else:
            spec = parse_color_spec(props)
        spec.apply(self.color_mapper)
        return


class ColorPaletteJS:
    """Client-side ColorPalette selector"""
    def __init__(self):
        self.widths = {
            "select": 140,
            "div": 300
        }
        # Map palettes to ColumnDataSource
        names, numbers = [], []
        for name, palettes in sorted(bokeh.palettes.all_palettes.items()):
            for number in sorted(palettes.keys()):
                names.append(name)
                numbers.append(number)
        self.source = bokeh.models.ColumnDataSource({
            "names": names,
            "numbers": numbers
        })

        # Figure to display color bar preview
        self.color_mapper = bokeh.models.LinearColorMapper(
            palette="Greys256",
            low=0,
            high=1)
        self.figure = colorbar_figure(self.color_mapper,
                                      plot_width=320)

        # Wire up select widgets
        self.selects = {
            "name": bokeh.models.Select(width=self.widths["select"]),
            "number": bokeh.models.Select(width=self.widths["select"]),
        }
        self.selects["name"].options = ["Please specify"] + list(sorted(set(names)))
        self.selects["name"].value = "Please specify"
        self.selects["number"].options = ["Please specify"]
        self.selects["number"].value = "Please specify"
        custom_js = bokeh.models.CustomJS(args=dict(
                source=self.source,
                select=self.selects["number"]), code="""
            let name = cb_obj.value
            let names = source.data["names"]
            let numbers = source.data["numbers"]
            let options = ["Please specify"]
            for (let i=0; i<names.length; i++) {
                if (names[i] == name) {
                    options.push(numbers[i].toString())
                }
            }
            select.options = options
        """)
        self.selects["name"].js_on_change("value", custom_js)


        # Preview figure
        self.selects["name"].on_change("value", self.on_preview)
        self.selects["number"].on_change("value", self.on_preview)

        # Reverse checkbox
        self.checkboxes = {}
        self.checkboxes["reverse"] = bokeh.models.CheckboxGroup(
            labels=["Reverse"],
            active=[])
        self.checkboxes["reverse"].on_change("active", self.on_preview)

        self.layout = bokeh.layouts.column(
            bokeh.models.Div(text="Color palette:",
                             width=self.widths["div"]),
            self.figure,
            bokeh.layouts.row(
                self.selects["name"],
                self.selects["number"]),
            self.checkboxes["reverse"])

    def on_preview(self, attr, old, new):
        spec = ColorSpec(**self.props())
        try:
            spec.apply(self.color_mapper)
        except KeyError:
            pass

    def props(self):
        """Useful for aggregating form data"""
        _props = {}
        for key, select in self.selects.items():
            if select.value is not None:
                _props[key] = select.value
        _props["reverse"] = len(self.checkboxes["reverse"].active) == 1
        return _props

    def render(self, props):
        for key, select in self.selects.items():
            if key in props:
                select.value = str(props[key])
        self.checkboxes["reverse"].active = {
            True: [0],
            False: []
        }[props.get("reverse", False)]


class ColorPalette(Observable):
    """Color palette user interface"""
    def __init__(self):
        self.dropdowns = {
            "names": bokeh.models.Dropdown(label="Palettes"),
            "numbers": bokeh.models.Dropdown(label="N")
        }
        self.dropdowns["names"].on_click(self.on_name)
        self.dropdowns["numbers"].on_click(self.on_number)

        self.checkbox = bokeh.models.CheckboxGroup(
            labels=["Reverse"],
            active=[])
        self.checkbox.on_change("active", self.on_reverse)

        self.layout = bokeh.layouts.column(
                bokeh.models.Div(text="Color palette:"),
                self.dropdowns["names"],
                self.dropdowns["numbers"],
                self.checkbox)
        super().__init__()

    def connect(self, store):
        """Connect component to Store"""
        connect(self, store)
        return self

    def props(self):
        """Helper to get widget settings"""
        return {
            "name": self.dropdowns["names"].label,
            "number": self.dropdowns["numbers"].label,
            "reverse": len(self.checkbox.active) == 1
        }

    def on_name(self, event):
        """Event-handler when a palette name is selected"""
        self.notify(set_palette_name(event.item))

    def on_number(self, event):
        """Event-handler when a palette number is selected"""
        self.notify(set_palette_number(int(event.item)))

    def on_reverse(self, attr, old, new):
        """Event-handler when reverse toggle is changed"""
        self.notify(set_reverse(len(new) == 1))

    def render(self, props):
        """Render component from properties derived from state"""
        assert isinstance(props, dict), "only support dict"

        spec = parse_color_spec(props)
        if "name" in props:
            self.dropdowns["names"].label = spec.name
        if "number" in props:
            self.dropdowns["numbers"].label = str(spec.number)

        if "names" in props:
            values = props["names"]
            self.dropdowns["names"].menu = list(zip(values, values))
        if "numbers" in props:
            values = [str(n) for n in props["numbers"]]
            self.dropdowns["numbers"].menu = list(zip(values, values))

        # Render checkbox state
        if spec.reverse:
            self.checkbox.active = [0]
        else:
            self.checkbox.active = []
