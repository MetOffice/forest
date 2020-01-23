import bokeh.plotting
from bokeh.plotting.helpers import _glyph_function
import bokeh.models
from bokeh.core.properties import DistanceSpec


class Barb(bokeh.models.XYGlyph):
    __implementation__ = "barb.ts"
    _args = ('x', 'y', 'u', 'v')
    x = DistanceSpec(units_default="screen")
    y = DistanceSpec(units_default="screen")
    u = DistanceSpec(units_default="screen")
    v = DistanceSpec(units_default="screen")


# Extend bokeh.plotting.Figure to support .barb()
if not hasattr(bokeh.plotting.Figure, 'barb'):
    bokeh.plotting.Figure.barb = _glyph_function(Barb)
