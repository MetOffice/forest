from abc import ABC, abstractmethod
import datetime as dt
import numpy as np
import bokeh.models
import forest.data
from forest import geo, colors
from forest.old_state import old_state, unique
from forest.exceptions import FileNotFound, IndexNotFound


def map_view(loader, color_mapper, use_hover_tool=True, tooltips=None):
    """Convenient method to simplify MapView construction"""
    if forest.data.FEATURE_FLAGS["multiple_colorbars"]:
        color_mapper = bokeh.models.LinearColorMapper(
            palette="Greys256",
            low=0,
            high=1)
        color_view = ColorView(color_mapper)
        image_view = ImageView(loader, color_mapper,
                               use_hover_tool=use_hover_tool)
        if tooltips is not None:
            image_view.tooltips = tooltips
        return MapView(image_view, color_view)
    else:
        view = ImageView(loader, color_mapper,
                         use_hover_tool=use_hover_tool)
        if tooltips is not None:
            view.tooltips = tooltips
        return view


class AbstractMapView(ABC):
    @abstractmethod
    def add_figure(self, figure):
        pass

    @abstractmethod
    def render(self, state):
        pass


class MapView(AbstractMapView):
    """Extend ImageView to support color_mapper"""
    def __init__(self, um_view, color_view):
        self.um_view = um_view
        self.color_view = color_view

    @property
    def image_sources(self):
        return self.um_view.image_sources

    def add_figure(self, figure):
        return self.um_view.add_figure(figure)

    def render(self, state):
        self.color_view.render(state)
        self.um_view.render(state)


class ColorView:
    """Applies ColorSpec to bokeh.models.LinearColorMapper"""
    def __init__(self, color_mapper):
        self.color_mapper = color_mapper

    def render(self, state):
        if "colorbar" in state:
            spec = colors.parse_color_spec(state["colorbar"])
            spec.apply(self.color_mapper)


class ImageView(AbstractMapView):
    def __init__(self, loader, color_mapper, use_hover_tool=True):
        self.loader = loader
        self.color_mapper = color_mapper
        self.color_mapper.nan_color = bokeh.colors.RGB(0, 0, 0, a=0)
        self.use_hover_tool = use_hover_tool
        self.source = bokeh.models.ColumnDataSource({
                "x": [],
                "y": [],
                "dw": [],
                "dh": [],
                "image": []})
        self.image_sources = [self.source]

        self.tooltips = [
            ("Name", "@name"),
            ("Value", "@image @units"),
            ('Length', '@length'),
            ('Valid', '@valid{%F %H:%M}'),
            ('Initial', '@initial{%F %H:%M}'),
            ("Level", "@level")]

        self.formatters = {
            '@valid': 'datetime',
            '@initial': 'datetime'
        }

    @old_state
    @unique
    def render(self, state):
        self.source.data = self.loader.image(state)

    def set_hover_properties(self, tooltips, formatters):
        self.tooltips = tooltips
        self.formatters = formatters

    def add_figure(self, figure):
        renderer = figure.image(
                x="x",
                y="y",
                dw="dw",
                dh="dh",
                image="image",
                source=self.source,
                color_mapper=self.color_mapper)
        if self.use_hover_tool:
            tool = bokeh.models.HoverTool(
                    renderers=[renderer],
                    tooltips=self.tooltips,
                    formatters=self.formatters)
            figure.add_tools(tool)
        return renderer
