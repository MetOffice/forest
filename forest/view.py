from abc import ABC, abstractmethod
import datetime as dt
import numpy as np
import bokeh.models
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
