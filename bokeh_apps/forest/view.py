import bokeh.models
import geo


class EarthNetworks(object):
    def __init__(self, loader):
        frame = loader.frame
        if frame is not None:
            x, y = geo.web_mercator(
                    frame.longitude,
                    frame.latitude)
            date = frame.date
            longitude = frame.longitude
            latitude = frame.latitude
            flash_type = frame.flash_type
        else:
            x, y = [], []
            date = []
            longitude = []
            latitude = []
            flash_type = []
        self.source = bokeh.models.ColumnDataSource({
            "x": x,
            "y": y,
            "date": date,
            "longitude": longitude,
            "latitude": latitude,
            "flash_type": flash_type,
        })

    def add_figure(self, figure):
        renderer = figure.circle(
                x="x",
                y="y",
                size=10,
                source=self.source)
        tool = bokeh.models.HoverTool(
                tooltips=[
                    ('Time', '@date{%F}'),
                    ('Lon', '@longitude'),
                    ('Lat', '@latitude'),
                    ('Flash type', '@flash_type')],
                formatters={
                    'date': 'datetime'
                },
                renderers=[renderer])
        figure.add_tools(tool)
        return renderer


class RDT(object):
    def __init__(self, loader):
        self.color_mapper = bokeh.models.CategoricalColorMapper(
                palette=bokeh.palettes.Spectral6,
                factors=["0", "1", "2", "3", "4"])
        self.source = bokeh.models.GeoJSONDataSource(
                geojson=loader.geojson)

    def add_figure(self, figure):
        renderer = figure.patches(
            xs="xs",
            ys="ys",
            fill_alpha=0,
            line_width=2,
            line_color={
                'field': 'PhaseLife',
                'transform': self.color_mapper},
            source=self.source)
        tool = bokeh.models.HoverTool(
                tooltips=[
                    ('CType', '@CType'),
                    ('CRainRate', '@CRainRate'),
                    ('ConvTypeMethod', '@ConvTypeMethod'),
                    ('ConvType', '@ConvType'),
                    ('ConvTypeQuality', '@ConvTypeQuality'),
                    ('SeverityIntensity', '@SeverityIntensity'),
                    ('MvtSpeed', '@MvtSpeed'),
                    ('MvtDirection', '@MvtDirection'),
                    ('NumIdCell', '@NumIdCell'),
                    ('CTPressure', '@CTPressure'),
                    ('CTPhase', '@CTPhase'),
                    ('CTReff', '@CTReff'),
                    ('LonG', '@LonG'),
                    ('LatG', '@LatG'),
                    ('ExpansionRate', '@ExpansionRate'),
                    ('BTmin', '@BTmin'),
                    ('BTmoy', '@BTmoy'),
                    ('CTCot', '@CTCot'),
                    ('CTCwp', '@CTCwp'),
                    ('NbPosLightning', '@NbPosLightning'),
                    ('SeverityType', '@SeverityType'),
                    ('Surface', '@Surface'),
                    ('Duration', '@Duration'),
                    ('CoolingRate', '@CoolingRate'),
                    ('Phase life', '@PhaseLife')],
                renderers=[renderer])
        figure.add_tools(tool)
        return renderer


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

    def render(self, variable, ipressure, itime):
        if variable is None:
            return
        self.source.data = self.loader.image(
                variable,
                ipressure,
                itime)

        def on_change(attr, old, new):
            print(attr, old, new)

        self.source.selected.on_change("indices",
                on_change)

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

    def render(self, variable, ipressure, itime):
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
