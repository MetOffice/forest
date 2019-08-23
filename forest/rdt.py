import os
import glob
import re
import datetime as dt
import bokeh
import json
import numpy as np
import geo
from util import timeout_cache


class View(object):
    def __init__(self, loader):
        self.loader = loader
        empty = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0]]]
                    },
                    "properties": {
                        'LatG': 0,
                        'LonG': 0,
                        'CType': 0,
                        'CRainRate': 0,
                        'CType': 0,
                        'CRainRate': 0,
                        'ConvTypeMethod': 0,
                        'ConvType': 0,
                        'ConvTypeQuality': 0,
                        'SeverityIntensity': 0,
                        'MvtSpeed': '',
                        'MvtDirection': '',
                        'NumIdCell': 0,
                        'CTPressure': 0,
                        'CTPhase': '',
                        'CTReff': '',
                        'ExpansionRate': '',
                        'BTmin': 0,
                        'BTmoy': 0,
                        'CTCot': '',
                        'CTCwp': '',
                        'NbPosLightning': 0,
                        'SeverityType': '',
                        'Surface': '',
                        'Duration': 0,
                        'CoolingRate': 0,
                        'PhaseLife': "0"
                    }
                }
            ]
        }
        self.empty_geojson = json.dumps(empty)
        print(self.empty_geojson)
        self.color_mapper = bokeh.models.CategoricalColorMapper(
                palette=bokeh.palettes.Spectral6,
                factors=["0", "1", "2", "3", "4"])
        self.source = bokeh.models.GeoJSONDataSource(
                geojson=self.empty_geojson)

    def render(self, state):
        if state.valid_time is not None:
            date = dt.datetime.strptime(state.valid_time, '%Y-%m-%d %H:%M:%S')
            print(date)
            self.source.geojson = self.loader.load_date(date)

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
    def __init__(self, pattern):
        self.pattern = pattern
        self.geojson = self.load(self.paths[0])
        super().__init__()

    @property
    def paths(self):
        return self.find(self.pattern)

    @staticmethod
    @timeout_cache(dt.timedelta(minutes=10))
    def find(pattern):
        return sorted(glob.glob(pattern))

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
        i = np.argmin(np.abs(self.dates(self.paths) - date))
        return self.paths[i]

    def dates(self, paths):
        return np.array([
            self.parse_date(p) for p in paths],
            dtype='datetime64[s]')

    def parse_date(self, path):
        groups = re.search(r"[0-9]{12}", os.path.basename(path))
        if groups is not None:
            return dt.datetime.strptime(groups[0], "%Y%m%d%H%M")
