import bokeh.models
import geo


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
                    ("Value", "@image"),
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

    def image(self, time):
        print("EIDA50: {}".format(time))
        self.source.data = self.loader.image(time)

    def add_figure(self, figure):
        return figure.image(
                x="x",
                y="y",
                dw="dw",
                dh="dh",
                image="image",
                source=self.source,
                color_mapper=self.color_mapper)
