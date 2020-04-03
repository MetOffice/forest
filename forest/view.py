from abc import ABC, abstractmethod
import datetime as dt
import numpy as np
import bokeh.models
import forest.colors
from forest import geo
from forest.old_state import old_state, unique
from forest.exceptions import FileNotFound, IndexNotFound


class AbstractMapView(ABC):
    @abstractmethod
    def add_figure(self, figure):
        pass

    @abstractmethod
    def render(self, state):
        pass


def color_image(label, loader, use_hover_tool=True):
    """Helper to create a ColorImageView

    .. note:: This is convenient since there are
              multiple drivers that use color imagery
    """
    # ColorView
    color_mapper = bokeh.models.LinearColorMapper()
    color_view = forest.colors.ColorView(
        color_mapper,
        forest.colors.SpecParser(label))

    # ImageView
    image_view = UMView(
        loader,
        color_mapper,
        use_hover_tool=use_hover_tool)
    return ColorImageView(image_view, color_view)


class ColorImageView:
    """Delegates to more specialist views

    Facade to combine color_mapper and image glyph views
    behind a single interface
    """
    def __init__(self, image_view, color_view):
        self.image_view = image_view
        self.color_view = color_view

    @property
    def image_sources(self):
        # TODO: Remove this delegation by redesigning how source limits
        #       are calculated
        return self.image_view.image_sources

    def render(self, state):
        self.image_view.render(state)

    def connect(self, store):
        """Subscribe to store"""
        self.color_view.connect(store)
        self.image_view.connect(store)  # TODO: Check if this is needed
        return self

    def add_figure(self, figure):
        """Add glyphs to figure"""
        return self.image_view.add_figure(figure)


class UMView(AbstractMapView):
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
            'valid': 'datetime',
            'initial': 'datetime'
        }

    def connect(self, store):
        """Represent application state changes"""
        store.add_subscriber(self.render)
        return self

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


class Image(object):
    pass


class Barbs(object):
    pass


class NearCast(object):
    def __init__(self, loader, color_mapper):
        self.loader = loader
        self.color_mapper = color_mapper
        self.color_mapper.nan_color = bokeh.colors.RGB(0, 0, 0, a=0)
        self.source = bokeh.models.ColumnDataSource({
                "x": [],
                "y": [],
                "dw": [],
                "dh": [],
                "image": []})
        self.image_sources = [self.source]

    @old_state
    @unique
    def render(self, state):
        self.source.data = self.loader.image(state)

    def set_hover_properties(self, tooltips):
        self.tooltips = tooltips

    def add_figure(self, figure):
        renderer = figure.image(
                   x="x",
                   y="y",
                   dw="dw",
                   dh="dh",
                   image="image",
                   source=self.source,
                   color_mapper=self.color_mapper)

        tool = bokeh.models.HoverTool(
               renderers=[renderer],
               tooltips=self.tooltips)

        figure.add_tools(tool)
        return renderer
