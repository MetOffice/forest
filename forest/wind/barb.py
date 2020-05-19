import bokeh.plotting
import bokeh.models
from bokeh.core.properties import DistanceSpec

class Barb(bokeh.models.XYGlyph):
    __implementation__ = "barb.ts"
    _args = ('x', 'y', 'u', 'v')
    x = DistanceSpec(units_default="screen")
    y = DistanceSpec(units_default="screen")
    u = DistanceSpec(units_default="screen")
    v = DistanceSpec(units_default="screen")

def barb():
    '''Dummy function to be replaced with decorated function'''
    return True


# Extend bokeh.plotting.Figure to support .barb()
if not hasattr(bokeh.plotting.Figure, 'barb'):
    bokeh.plotting.Figure.barb = bokeh.plotting._decorators.glyph_method(Barb)(barb)
