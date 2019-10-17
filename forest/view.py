import datetime as dt
import numpy as np
import bokeh.models
from forest import geo
from forest.exceptions import FileNotFound, IndexNotFound


class UMView(object):
    def __init__(self, loader, color_mapper):
        self.loader = loader
        self.color_mapper = color_mapper
        self.source = bokeh.models.ColumnDataSource({
                "x": [],
                "y": [],
                "dw": [],
                "dh": [],
                "image": []})

    def render(self, state):
        self.source.data = self.loader.image(state)

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
                tooltips=[
                    ("Name", "@name"),
                    ("Value", "@image @units"),
                    ('Length', '@length'),
                    ('Valid', '@valid{%F %H:%M}'),
                    ('Initial', '@initial{%F %H:%M}'),
                    ("Level", "@level")],
                formatters={
                    'valid': 'datetime',
                    'initial': 'datetime'
                })
        figure.add_tools(tool)
        return renderer


class Image(object):
    pass


class Barbs(object):
    pass


class GPMView(object):
    def __init__(self, loader, color_mapper):
        self.loader = loader
        self.color_mapper = color_mapper
        self.empty = {
                "lons": [],
                "lats": [],
                "x": [],
                "y": [],
                "dw": [],
                "dh": [],
                "image": []}
        self.source = bokeh.models.ColumnDataSource(self.empty)

    def render(self, variable, pressure, itime):
        if variable != "precipitation_flux":
            self.source.data = self.empty
        else:
            self.source.data = self.loader.image(itime)

    def add_figure(self, figure):
        return figure.image(
                x="x",
                y="y",
                dw="dw",
                dh="dh",
                image="image",
                source=self.source,
                color_mapper=self.color_mapper)

class EIDA50(object):
    def __init__(self, loader, color_mapper):
        self.loader = loader
        self.color_mapper = color_mapper
        self.empty = {
                "x": [],
                "y": [],
                "dw": [],
                "dh": [],
                "image": []}
        self.source = bokeh.models.ColumnDataSource(
                self.empty)

    def render(self, state):
        if state.valid_time is not None:
            self.image(self.to_datetime(state.valid_time))

    @staticmethod
    def to_datetime(d):
        if isinstance(d, dt.datetime):
            return d
        elif isinstance(d, str):
            try:
                return dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return dt.datetime.strptime(d, "%Y-%m-%dT%H:%M:%S")
        elif isinstance(d, np.datetime64):
            return d.astype(dt.datetime)
        else:
            raise Exception("Unknown value: {}".format(d))

    def image(self, time):
        try:
            self.source.data = self.loader.image(time)
        except (FileNotFound, IndexNotFound):
            self.source.data = self.empty

    def add_figure(self, figure):
        return figure.image(
                x="x",
                y="y",
                dw="dw",
                dh="dh",
                image="image",
                source=self.source,
                color_mapper=self.color_mapper)
