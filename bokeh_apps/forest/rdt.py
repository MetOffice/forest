import os
import re
import datetime as dt
import bokeh
import json
import numpy as np
import geo


class View(object):
    def __init__(self, loader):
        self.loader = loader
        self.color_mapper = bokeh.models.CategoricalColorMapper(
                palette=bokeh.palettes.Spectral6,
                factors=["0", "1", "2", "3", "4"])
        self.source = bokeh.models.GeoJSONDataSource(
                geojson=loader.geojson)

    def render(self, valid_date):
        self.source.geojson = self.loader.load_date(valid_date)

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


class Loader(object):
    def __init__(self, paths):
        self.paths = paths
        self.dates = np.array([self.parse_date(p) for p in paths], dtype='datetime64[s]')
        self.geojson = self.load(self.paths[0])

    def load_date(self, date):
        return self.load(self.find_file(date))

    @staticmethod
    def load(path):
        print(path)
        with open(path) as stream:
            rdt = json.load(stream)

        copy = dict(rdt)
        for i, feature in enumerate(rdt["features"]):
            coordinates = feature['geometry']['coordinates'][0]
            lons, lats = np.asarray(coordinates).T
            x, y = geo.web_mercator(lons, lats)
            c = np.array([x, y]).T.tolist()
            copy["features"][i]['geometry']['coordinates'][0] = c

        # Hack to use Categorical mapper
        for i, feature in enumerate(rdt["features"]):
            p = feature['properties']['PhaseLife']
            copy["features"][i]['properties']['PhaseLife'] = str(p)

        return json.dumps(copy)

    def find_file(self, date):
        if isinstance(date, dt.datetime):
            date = np.datetime64(date, 's')
        i = np.argmin(np.abs(self.dates - date))
        return self.paths[i]

    def parse_date(self, path):
        groups = re.search(r"[0-9]{12}", os.path.basename(path))
        if groups is not None:
            return dt.datetime.strptime(groups[0], "%Y%m%d%H%M")
