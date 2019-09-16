import os
import glob
import re
import datetime as dt
import bokeh
import json
import numpy as np
import geo
import locate
from util import timeout_cache
from exceptions import FileNotFound


class RenderGroup(object):
    def __init__(self, renderers, visible=False):
        self.renderers = renderers
        self.visible = visible

    @property
    def visible(self):
        return any(r.visible for r in self.renderers)

    @visible.setter
    def visible(self, value):
        for r in self.renderers:
            r.visible = value



class View(object):
    def __init__(self, loader):
        self.loader = loader
        ## TODO Need to put in more empty features (for lines, points etc) here as I add them to the loader
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

        empty_tail = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[[0, 0]]]
                    },
                    "properties": {
                        'NumIdCell': 0,
                        'NumIdBirth': 0
                    }
                }
            ]
        }
        self.empty_tail_line_source = json.dumps(empty_tail)
        self.empty_geojson = json.dumps(empty)

        # print(self.empty_geojson)
        self.color_mapper = bokeh.models.CategoricalColorMapper(
                palette=bokeh.palettes.Spectral6,
                factors=["0", "1", "2", "3", "4"])
        ## TODO setup new ColumnDataSource here for other objects (tails, other polygons and points)
        self.source = bokeh.models.GeoJSONDataSource(geojson=self.empty_geojson)
        self.tail_line_source = bokeh.models.GeoJSONDataSource(geojson=self.empty_tail_line_source)
        # self.tail_point_source = bokeh.models.ColumnDataSource(dict(x=[], y=[]))

    # Gets called when a menu button is clicked (or when application state changes)
    def render(self, state):
        print(state.valid_time)
        if state.valid_time is not None:
            date = dt.datetime.strptime(state.valid_time, '%Y-%m-%d %H:%M:%S')
            print('This is:',date)
            try:
                ## TODO Renderer changes data depending on date
                self.source.geojson, self.tail_line_source.geojson = self.loader.load_date(date)
                print(self.tail_line_source.geojson)
                # print(self.source.geojson)
                #self.circle_source.data = {'x': [1,2,3], 'y':[4,5,6]} # Need to put in smthg like self.tail_loader.load_date(date)
            except FileNotFound:
                print('File not found - bug?')
                self.source.geojson = self.empty_geojson
                self.tail_line_source.geojson = self.empty_tail_line_source

    def add_figure(self, figure):
        # This is where all the plotting happens (e.g. when the applciation is loaded)
        ## TODO Add in multi-line plotting method for the tails, polygons and points
        print('Adding figure')

        # circles = figure.circle(x=[3450904, 3651279], y=[-111325, -144727], size=5)
        lines = figure.multi_line(xs="xs", ys="ys", source=self.tail_line_source)
        #lines = figure.line(x=[3450904, 3651279], y=[-111325, -144727], line_width=3) # bokeh markers for more
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
                    ('Cloud Type', '@CType'),
                    ('Convective Rainfall Rate', '@CRainRate'),
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
                    ('Phase life', '@PhaseLife')], ## TODO Convert to human readable results. This will involve adding a PhaseLifeLabel to the geojson file
                renderers=[renderer])
        figure.add_tools(tool)
        return RenderGroup([renderer, lines, circles])

class TailLoader(object):

    ## TODO: Add another way of reading the geojson so that we can add in tails (replicate the Loader class) and copy again for other data types (such as polygons, points etc)
    # If I get stuck, I could write this as a function

    def __init__(self, locator):
        self.locator = locator # Locator(pattern)

    def load_date(self, date):
        return self.load(self.locator.find_file(date))

    @staticmethod
    def load(path):
        print(path)

        pt_fields = ['NumIdCell', 'NumIdBirth', 'DTimeTraj', 'LatTrajCellCG', 'LonTrajCellCG', 'BTempTraj', 'BTminTraj', 'BaseAreaTraj', 'TopAreaTraj', 'CoolingRateTraj', 'ExpanRateTraj', 'SpeedTraj', 'DirTraj']
        line_fields = ['NumIdCell', 'NumIdBirth']

        with open(path) as stream:
            rdt = json.load(stream)

        copy = dict(rdt)
        cds = bokeh.models.ColumnDataSource(dict(xs=[], ys=[], NumIdCell=[], NumIdBirth=[]))
        xs, ys = [], []
        for i, feature in enumerate(rdt["features"]):
            lons = feature['properties']['LonTrajCellCG']
            lats = feature['properties']['LatTrajCellCG']
            x, y = geo.web_mercator(lons, lats)
            c = np.array([x, y]).T.tolist() # Conveniently, a LineString has the same structure as a MultiPoint
            xs.append(x)
            ys.append(y)
            copy["features"][i]['geometry'] = {'type': 'LineString', 'coordinates': c}
            # copy["features"][i]['geometries'] = [ {'type': 'LineString', 'coordinates': c}, {'type': 'MultiPoint', 'coordinates': c} ]

        # Hack to use Categorical mapper
        # ## TODO Add in PhaseLifeLabel to properties dictionary
        # for i, feature in enumerate(rdt["features"]):
        #     p = feature['properties']['PhaseLife']
        #     copy["features"][i]['properties']['PhaseLife'] = fieldValueLUT('PhaseLife', p)

        print('Line load complete')
        # print([feat["geometry"]['coordinates'] for feat in copy["features"]])
        return json.dumps(copy)
        # return {"xs": xs, "ys": ys}

# def example_bokeh():
#     source = bokeh.models.ColumnDataSource({"x": [])
#     figure.circle(x="x", fill_color="BTmin", source=source, color_mapper=)

class Loader(object):
    def __init__(self, pattern):
        self.locator = Locator(pattern)
        self.poly_loader = PolygonLoader(self.locator)
        self.tail_loader = TailLoader(self.locator)


    def load_date(self, date):
        geojson_poly = self.poly_loader.load_date(date)
        geojson_tail = self.tail_loader.load_date(date)
        return [geojson_poly, geojson_tail]


class PolygonLoader(object):
    # Keep this loader as it is, and write loader for each RDT data type

    def __init__(self, locator):
        self.locator = locator # Locator(pattern)

    def load_date(self, date):
        return self.load(self.locator.find_file(date), date)

    @staticmethod
    def load(path, date):
        print(path)
        print(date)

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
        ## TODO Add in PhaseLifeLabel to properties dictionary
        for i, feature in enumerate(rdt["features"]):
            p = feature['properties']['PhaseLife']
            copy["features"][i]['properties']['PhaseLifeLabel'] = fieldValueLUT('PhaseLife', p)
            copy["features"][i]['properties']['PhaseLife'] = str(p)
            # print(p, fieldValueLUT('PhaseLife', p))

        print('Polygon load complete')
        return json.dumps(copy)


def rdtUnits(fn, data):
    # Converts units according to netcdf files definition
    rdtUnitsLUT = {
        'DecTime': {'scale': 1.0, 'offset': 0.0, 'Units': 's'},
        'LeadTime': {'scale': 1.0, 'offset': 0.0, 'Units': 's'},
        'Duration': {'scale': 1.0, 'offset': 0.0, 'Units': 's'},
        'MvtSpeed': {'scale': 0.001, 'offset': 0.0, 'Units': 'm s-1'},
        'MvtDirection': {'scale': 1.0, 'offset': 0.0, 'Units': 'degree'},
        'DtTimeRate': {'scale': 1.0, 'offset': 0.0, 'Units': 's'},
        'ExpansionRate': {'scale': 2e-07, 'offset': -0.005, 'Units': 's-1'},
        'CoolingRate': {'scale': 2e-06, 'offset': -0.05, 'Units': 'K s-1'},
        'LightningRate': {'scale': 1e-04, 'offset': -2.0, 'Units': 's-1'},
        'CTPressRate': {'scale': 0.001, 'offset': -25.0, 'Units': 'Pa s-1'},
        'BTemp': {'scale': 0.01, 'offset': 130.0, 'Units': 'K'},
        'BTmoy': {'scale': 0.01, 'offset': 130.0, 'Units': 'K'},
        'BTmin': {'scale': 0.01, 'offset': 130.0, 'Units': 'K'},
        'Surface': {'scale': 5000000.0, 'offset': 0.0, 'Units': 'm2'},
        'EllipseGaxe': {'scale': 20.0, 'offset': 0.0, 'Units': 'm'},
        'EllipsePaxe': {'scale': 20.0, 'offset': 0.0, 'Units': 'm'},
        'EllipseAngle': {'scale': 1.0, 'offset': 0.0, 'Units': 'degrees_north'},
        'DtLightning': {'scale': 1.0, 'offset': 0.0, 'Units': 's'},
        'CTPressure': {'scale': 10.0, 'offset': 0.0, 'Units': 'Pa'},
        'CTCot': {'scale': 0.01, 'offset': 0.0, 'Units': '1'},
        'CTReff': {'scale': 1e-08, 'offset': 0.0, 'Units': 'm'},
        'CTCwp': {'scale': 0.001, 'offset': 0.0, 'Units': 'kg m-2'},
        'CRainRate': {'scale': 0.1, 'offset': 0.0, 'Units': 'mm/h'},
        'BTempSlice': {'scale': 0.01, 'offset': 130.0, 'Units': 'K'},
        'SurfaceSlice': {'scale': 5000000.0, 'offset': 0.0, 'Units': 'm2'},
        'DTimeTraj': {'scale': 1.0, 'offset': 0.0, 'Units': 's'},
        'BTempTraj': {'scale': 0.01, 'offset': 130.0, 'Units': 'K'},
        'BTminTraj': {'scale': 0.01, 'offset': 130.0, 'Units': 'K'},
        'BaseAreaTraj': {'scale': 5000000.0, 'offset': 0.0, 'Units': 'm2'},
        'TopAreaTraj': {'scale': 5000000.0, 'offset': 0.0, 'Units': 'm2'},
        'CoolingRateTraj': {'scale': 2e-06, 'offset': -0.05, 'Units': 'K s-1'},
        'ExpanRateTraj': {'scale': 2e-07, 'offset': -0.005, 'Units': 's-1'},
        'SpeedTraj': {'scale': 0.001, 'offset': 0.0, 'Units': 'm s-1'},
        'DirTraj': {'scale': 1.0, 'offset': 0.0, 'Units': 'degree'}
    }

    dict = rdtUnitsLUT.get(fn, {'scale': 1.0, 'offset': 1.0, 'units': '-'})
    scale, offset, units = dict.values()

    conv_data = ( data / scale ) + offset

    return(conv_data, units)


def fieldNameLUT(fn):

    short2long = {
                    'NbSigCell': 'Number of encoded significant RDT-CW cloud cells',
                    'NbConvCell': 'Number of convective cloud cells',
                    'NbCloudCell': 'Number of analyzed and tracked cloud cells',
                    'NbElecCell': 'Number of electric cloud cells',
                    'NbHrrCell': 'Number of cloud cells with High Rain Rate values ',
                    'MapCellCatType': 'NWC GEO RDT-CW Type and phase of significant cells',
                    'NumIdCell': 'Identification Number of cloud cell',
                    'NumIdBirth': 'Identification Number of cloud cell at birth',
                    'DecTime': 'time gap between radiometer time and slot time',
                    'LeadTime': 'lead time from slot for forecast cloud cell',
                    'Duration': 'Duration of cloud system since birth',
                    'ConvType': 'Type (conv or not) of cloud system',
                    'ConvTypeMethod': 'Method used for convective diagnosis',
                    'ConvTypeQuality': 'Quality of convective diagnosis ',
                    'PhaseLife': 'Phase Life of Cloud system',
                    'MvtSpeed': 'Motion speed of cloud cell',
                    'MvtDirection': 'Direction of Motion of cloud cell',
                    'MvtQuality': 'Quality of motion estimation of cloud cell',
                    'DtTimeRate': 'gap time to compute rates of cloud system',
                    'ExpansionRate': 'Expansion rate of cloud system',
                    'CoolingRate': 'Temperature change rate of cloud system',
                    'LightningRate': 'Lightning trend of cloud system',
                    'CTPressRate': 'Top Pressure trend of cloud system',
                    'SeverityType': 'Type of severity of cloud cell',
                    'SeverityIntensity': 'severity intensity of cloud cell ',
                    'LatContour': 'latitude of contour point of cloud cell',
                    'LonContour': 'longitude of contour point of cloud cell',
                    'LatG': 'latitude of Gravity Centre of cloud cell',
                    'LonG': 'longitude of Gravity Centre of cloud cell',
                    'BTemp': 'Brightness Temperature threshold defining a Cloud cell',
                    'BTmoy': 'Average Brightness Temperature over a Cloud cell',
                    'BTmin': 'Minimum Brightness Temperature of a Cloud cell',
                    'Surface': 'Surface of a Cloud cell',
                    'EllipseGaxe': 'Large axis of Ellipse approaching Cloud cell',
                    'EllipsePaxe': 'Small axis of Ellipse approaching Cloud cell',
                    'EllipseAngle': 'Angle of Ellipse approaching Cloud cell',
                    'NbPosLightning': 'Number of CG positive lightning strokes paired with cloud cell',
                    'NbNegLightning': 'Number of CG negative lightning strokes paired with cloud cell',
                    'NbIntraLightning': 'Number of IntraCloud lightning strokes paired with cloud cell',
                    'DtLightning': 'time interval to pair lighting data with cloud cells',
                    'CType': 'Most frequent Cloud Type over cloud cell extension',
                    'CTPhase': 'Most frequent Cloud Top Phase over cloud cell extension',
                    'CTPressure': 'Minimum Cloud Top Pressure over cloud cell extension',
                    'CTCot': 'maximum cloud_optical_thickness over cloud cell extension',
                    'CTReff': 'maximum radius_effective over cloud cell extension',
                    'CTCwp': 'maximum cloud condensed water_path over cloud cell extension',
                    'CTHicgHzd': 'High altitude Icing Hazard index',
                    'CRainRate': 'maximum convective_rain_rate over cloud cell extension',
                    'BTempSlice': 'Brightness Temperature threshold defining a Cloud cell',
                    'SurfaceSlice': 'Surface of Cloud cell at Temperature threshold',
                    'DTimeTraj': 'time gap between current and past Cloud cell',
                    'LatTrajCellCG': 'latitude of Gravity Centre of past cloud cell',
                    'LonTrajCellCG': 'longitude of Gravity Centre of past cloud cell',
                    'BTempTraj': 'Brightness Temperature threshold defining past Cloud cell',
                    'BTminTraj': 'Minimum Brightness Temperature of past Cloud cell',
                    'BaseAreaTraj': 'Surface of base of past Cloud cell',
                    'TopAreaTraj': 'Surface of top of past Cloud cell',
                    'CoolingRateTraj': 'Temperature change rate of past cloud system',
                    'ExpanRateTraj': 'Expansion rate of past cloud system',
                    'SpeedTraj': 'Motion speed of past cloud cell',
                    'DirTraj': 'Direction of Motion of past cloud cell',
                    'lat': 'Latitude at the centre of each pixel',
                    'lon': 'Longitude at the centre of each pixel',
                    'ny': 'Y Georeferenced Coordinate for each pixel count',
                    'nx': 'X Georeferenced Coordinate for each pixel count',
                    'MapCellCatType_pal': 'RGB palette for MapCellCatType'
    }
    try:
        return short2long.get(fn)
    except:
        return fn

def fieldValueLUT(fn, uid):

    rdtLUT = {
        "MapCellCatType": {
            0: "Non convective",
            1: "Convective triggering",
            2: "Convective triggering from split",
            3: "Convective growing",
            4: "Convective mature",
            5: "OvershootingTop mature",
            6: "Convective decaying",
            7: "Electric triggering",
            8: "Electric triggering from split",
            9: "Electric growing",
            10: "Electric mature",
            11: "Electric decaying",
            12: "HighRainRate triggering",
            13: "HighRainRate triggering from split",
            14: "HighRainRate growing",
            15: "HighRainRate mature",
            16: "HighRainRate decaying",
            17: "HighSeverity triggering",
            18: "HighSeverity triggering from split",
            19: "HighSeverity growing",
            20: "HighSeverity mature",
            21: "HighSeverity decaying"
        },
        'ConvType': {
            0: "Non convective",
            1: "Convective",
            2: "Convective inherited",
            3: "Convective forced overshoot",
            4: "Convective forced lightning",
            5: "Convective forced convrainrate",
            6: "Convective forced coldtropical",
            7: "Convective forced inherited",
            8: "Declassified convective",
            9: "Not defined"
        },
        'ConvTypeMethod': {
            1: "discrimination statistical scheme",
            2: "electric",
            3: "overshoot",
            4: "convective rain rate",
            5: "tropical"
        },
        'ConvTypeQuality': {
            1: "High quality",
            2: "Moderate quality",
            3: "Low quality",
            4: "Very low quality"
        },
        'PhaseLife': {
            0: "Triggering",
            1: "Triggering from split",
            2: "Growing",
            3: "Mature",
            4: "Decaying"
        },
        'MvtQuality': {
            1: "High quality",
            2: "Moderate quality",
            3: "Low quality",
            4: "Very low quality"
        },
        'SeverityType' : {
            0: "no activity",
            1: "turbulence",
            2: "lightning",
            3: "icing",
            4: "high altitude icing",
            5: "hail",
            6: "heavy rainfall",
            7: "type not defined"
        },
        'SeverityIntensity' : {
            0: "severity not defined",
            1: "low severity",
            2: "moderate severity",
            3: "high severity",
            4: "very high severity"
        },
        'CType' : {
            1: "Cloud-free land",
            2: "Cloud-free sea",
            3: "Snow over land",
            4: "Sea ice",
            5: "Very low clouds",
            6: "Low clouds",
            7: "Mid-level clouds",
            8: "High opaque clouds",
            9: "Very high opaque clouds",
            10: "Fractional clouds",
            11: "High semitransparent thin clouds",
            12: "High semitransparent meanly thick clouds",
            13: "High semitransparent thick clouds",
            14: "High semitransparent above low or medium clouds",
            15: "High semitransparent above snow ice"
        },
        'CTPhase' : {
            1: "Liquid",
            2: "Ice",
            3: "Mixed",
            4: "Cloud-free",
            5: "Undefined separability problems"
        },
        'CTHicgHzd' : {
            0: "Risk not defined",
            1: "low risk",
            2: "moderate risk",
            3: "high risk-free",
            4: "very high risk"
        }
    }

    try:
        return rdtLUT.get(fn)[uid]
    except:
        return "-"


class Locator(object):
    def __init__(self, pattern):
        self.pattern = pattern
        print(pattern)

    def find_file(self, valid_date):
        paths = np.array(self.paths)  # Note: timeout cache in use
        print(paths)
        bounds = locate.bounds(
                self.dates(paths),
                dt.timedelta(minutes=15))
        pts = locate.in_bounds(bounds, valid_date)
        found = paths[pts]
        if len(found) > 0:
            return found[0]
        else:
            raise FileNotFound("RDT: '{}' not found in {}".format(valid_date, bounds))

    @property
    def paths(self):
        return self.find(self.pattern)

    @staticmethod
    @timeout_cache(dt.timedelta(minutes=10))
    def find(pattern):
        return sorted(glob.glob(pattern))

    def dates(self, paths):
        return np.array([
            self.parse_date(p) for p in paths],
            dtype='datetime64[s]')

    def parse_date(self, path):
        groups = re.search(r"[0-9]{12}", os.path.basename(path))
        if groups is not None:
            return dt.datetime.strptime(groups[0], "%Y%m%d%H%M")
