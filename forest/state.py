"""
Application state
-----------------

Although the general design pattern of a redux store is defined by pure
reducer functions, engineers need documentation to extend and understand
:class:`State`.

Expressing state as a nested hierarchy of types allows readers of the code
to understand how data is organised. It also allows for type-checking to
simplify functions that manipulate state.

"""
import datetime as dt
from dataclasses import dataclass, field


@dataclass
class Limits:
    """
    Color map extent, high and low represent upper and lower limits
    respectively

    :param low: lower limit
    :type low: float
    :param high: upper limit
    :type high: float
    """
    low: float = 0.
    high: float = 1.


@dataclass
class ColorbarLimits:
    """Define user and column data source limits

    :param origin: either 'user' or 'column_data_source'
    :type origin: str
    :param column_data_source: column_data_source limits
    :type column_data_source: Limits
    :param user: user limits
    :type user: Limits
    """
    origin: str = "column_data_source"
    column_data_source: Limits = field(default_factory=Limits)
    user: Limits = field(default_factory=Limits)

    def __post_init__(self):
        if isinstance(self.column_data_source, dict):
            self.column_data_source = Limits(**self.column_data_source)
        if isinstance(self.user, dict):
            self.user = Limits(**self.user)


@dataclass
class Colorbar:
    """Colorbar settings

    :param name: bokeh palette name
    :param number: bokeh palette number
    :param limits: user and column_data_source limits
    :type limits: ColorbarLimits
    :param reverse: reverse color palette order
    :param invisible_min: hide/show values below minimum
    :param invisible_max: hide/show values above maximum
    """
    name: str = ""
    names: list = field(default_factory=list)
    number: int = 0
    numbers: list = field(default_factory=list)
    limits: ColorbarLimits = field(default_factory=ColorbarLimits)
    low: float = 0.
    high: float = 1.
    reverse: bool = False
    invisible_min: bool = False
    invisible_max: bool = False

    def __post_init__(self):
        if isinstance(self.limits, dict):
            self.limits = ColorbarLimits(**self.limits)


@dataclass
class LayerMode:
    """User-interface layout data

    :param state: Edit mode, either 'edit' or 'add'
    :param index: Index of layer being edited
    """
    state: str = ""
    index: int = 0


@dataclass
class Layers:
    """Layer-specific settings

    :param index: Map layer index to settings
    :param active: List of active layers
    :param mode: Edit/new mode to define UI
    :type mode: LayerMode
    """
    index: dict = field(default_factory=dict)
    active: list = field(default_factory=list)
    mode: LayerMode = field(default_factory=LayerMode)

    def __post_init__(self):
        if isinstance(self.mode, dict):
            self.mode = LayerMode(**self.mode)


@dataclass
class Tile:
    """Web map tiling user-settings

    :param name: Keyword to specify WMTS source
    :type name: str
    :param labels: Turn overlay labels on/off
    :type labels: bool
    """
    name: str = ""
    labels: bool = False


@dataclass
class Tools:
    """Flags to specify active tools

    :param time_series: Turn time series widget on/off
    :param profile: Turn profile widget on/off
    """
    time_series: bool = False
    profile: bool = False


@dataclass
class State:
    """Application State container

    Nested data structure to define components, views
    and behaviour.

    :param pattern: User-selected value
    :param patterns: Dataset-specific values
    :param variable: User-selected value
    :param variables: Dataset-specific values
    :param initial_time: User-selected value
    :param initial_times: Dataset-specific values
    :param valid_time: User-selected value
    :param valid_times: Dataset-specific values
    :param pressure: User-selected value
    :param pressures: Dataset-specific values
    :param layers: Layer-specific settings
    :type layers: Layers
    :param colorbar: Color mapper controls
    :type colorbar: Colorbar
    :param tile: Web map tiling configuration
    :type tile: Tile
    :param tools: Turn profile/time_series on/off
    :type tools: Tools
    """

    pattern: str = ""
    patterns: list = field(default_factory=list)
    variable: str = ""
    variables: list = field(default_factory=list)
    initial_time: dt.datetime = dt.datetime(1970, 1, 1)
    initial_times: list = field(default_factory=list)
    valid_time: dt.datetime = dt.datetime(1970, 1, 1)
    valid_times: list = field(default_factory=list)
    pressures: list = field(default_factory=list)
    colorbar: Colorbar = field(default_factory=Colorbar)
    layers: Layers = field(default_factory=Layers)
    dimension: dict = field(default_factory=dict)
    tile: Tile = field(default_factory=Tile)
    tools: Tools = field(default_factory=Tools)

    def __post_init__(self):
        """Type-checking"""
        if isinstance(self.colorbar, dict):
            self.colorbar = Colorbar(**self.colorbar)
        if isinstance(self.tile, dict):
            self.tile = Tile(**self.tile)
        if isinstance(self.tools, dict):
            self.tools = Tools(**self.tools)
        if isinstance(self.layers, dict):
            self.layers = Layers(**self.layers)

    @classmethod
    def from_dict(cls, data):
        """Factory method to simplify conversion from dict to dataclass"""
        return cls(**data)
