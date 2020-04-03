"""
Rapidly Developing Thunderstorms (RDT)
--------------------------------------
"""
import os
import glob
import re
import datetime as dt
import bokeh
import json
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import cf_units
from geojson import Point, Polygon, Feature, FeatureCollection
from forest import (
        geo,
        locate)
from forest.old_state import old_state, unique
import forest.util
from forest.exceptions import FileNotFound
from bokeh.palettes import GnBu3, OrRd3
import itertools
import math


class Dataset:
    def __init__(self, pattern=None, **kwargs):
        self.pattern = pattern
        self.locator = Locator(pattern)

    def navigator(self):
        return Navigator(self.locator)

    def map_view(self):
        return View(Loader(self.pattern))


class Navigator:
    """Navigator API facade"""
    def __init__(self, locator):
        self.locator = locator

    def variables(self, *args, **kwargs):
        return ["RDT"]

    def initial_times(self, *args, **kwargs):
        return [dt.datetime(1970, 1, 1)]

    def valid_times(self, *args, **kwargs):
        return self.locator.valid_times()

    def pressures(self, *args, **kwargs):
        return []


class RenderGroup:
    """Collection of renderers that act as one"""
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


class View:
    """Rapidly Developing Thunderstorms (RDT) visualisation"""
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
                        'CType': 0,
                        'CRainRate': 0,
                        'ConvTypeMethod': 0,
                        'ConvType': 0,
                        'ConvTypeQuality': 0,
                        'SeverityIntensity': 0,
                        'MvtSpeed': 0,
                        'MvtDirection': 0,
                        'NumIdCell': 0,
                        'CTPressure': 0,
                        'CTPhase': '',
                        'CTReff': '',
                        'ExpansionRate': '-',
                        'BTmin': 0,
                        'BTmoy': 0,
                        'CTCot': '-',
                        'CTCwp': '-',
                        'NbPosLightning': 0,
                        'SeverityType': '',
                        'Surface': '',
                        'Duration': 0,
                        'CoolingRate': 0,
                        'PhaseLife': "Triggering"
                    }
                }
            ]
        }
        self.empty_geojson = json.dumps(empty)
        self.empty_tail_line = dict(
                xs=[], ys=[],
                LonTrajCellCG=[],
                LatTrajCellCG=[],
                NumIdCell=[],
                NumIdBirth=[],
                DTimeTraj=[],
                BTempTraj=[],
                BTminTraj=[],
                BaseAreaTraj=[],
                TopAreaTraj=[],
                CoolingRateTraj=[],
                ExpanRateTraj=[],
                SpeedTraj=[],
                DirTraj=[])
        self.empty_tail_point = dict(
                x=[], y=[],
                LonTrajCellCG=[],
                LatTrajCellCG=[],
                NumIdCell=[],
                NumIdBirth=[],
                DTimeTraj=[],
                BTempTraj=[],
                BTminTraj=[],
                BaseAreaTraj=[],
                TopAreaTraj=[],
                CoolingRateTraj=[],
                ExpanRateTraj=[],
                SpeedTraj=[],
                DirTraj=[])
        self.empty_centre_point = dict(
                x1=[], y1=[], x2=[], y2=[], xs=[], ys=[],
                Arrowxs=[],
                Arrowys=[],
                LonG=[],
                LatG=[],
                NumIdCell=[],
                NumIdBirth=[],
                MvtSpeed=[],
                MvtDirection=[])

        self.color_mapper = bokeh.models.CategoricalColorMapper(
                palette=['#fee8c8', '#fdbb84', '#e34a33', '#43a2ca', '#a8ddb5'],
                factors=["Triggering", "Triggering from split", "Growing", "Mature", "Decaying"])
        self.source = bokeh.models.GeoJSONDataSource(geojson=self.empty_geojson)
        self.tail_line_source = bokeh.models.ColumnDataSource(self.empty_tail_line)
        self.tail_point_source = bokeh.models.ColumnDataSource(self.empty_tail_point)
        self.centre_point_source = bokeh.models.ColumnDataSource(self.empty_centre_point)

    @old_state
    @unique
    def render(self, state):
        """Gets called when a menu button is clicked (or when application state changes)"""
        if state.valid_time is not None:
            date = forest.util.to_datetime(state.valid_time)
            try:
                (self.source.geojson,
                 self.tail_line_source.data,
                 self.tail_point_source.data,
                 self.centre_point_source.data) = self.loader.load_date(date)
            except FileNotFound:
                print("rdt.View.render caught FileNotFound", date)
                self.source.geojson = self.empty_geojson
                self.tail_line_source.data = self.empty_tail_line
                self.tail_point_source.data = self.empty_tail_point
                self.centre_point_source.data = self.empty_centre_point

    def add_figure(self, figure):
        """This is where all the plotting happens (e.g. when the applciation is loaded)"""
        circles = figure.circle(x="x", y="y", size=3, source=self.tail_point_source)
        cntr_circles = figure.circle_cross(x="x1", y="y1", size=10, line_color='black', fill_color=None, source=self.centre_point_source)
        future_lines = figure.multi_line(xs="xs", ys="ys", line_color='black', source=self.centre_point_source)
        lines = figure.multi_line(xs="xs", ys="ys", line_dash='dashed', source=self.tail_line_source)
        arrows = figure.patches(xs='Arrowxs', ys='Arrowys', fill_color='black', line_color='black', source=self.centre_point_source)
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
                    ('NumIdCell', '@NumIdCell'),
                    ('Duration (Since Birth)', '@Duration{00:00:00}'),
                    ('Phase life', '@PhaseLife'), # Categorical
                    ('Cloud Type', '@CType'), # Categorical
                    ('Convective Rainfall Rate', '@CRainRate{0.0}' + ' mm/hr'),
                    ('Cloud System', '@ConvType'), # Categorical
                    ('Severity Type', '@SeverityType'), # Categorical
                    ('Severity Intensity', '@SeverityIntensity'), # Categorical
                    ('Cloud Top Phase', '@CTPhase'), # Categorical
                    ('Min. Cloud Top Pressure', '@CTPressure' + ' hPa'),
                    # ('Max. Cloud Top Effective Radius', '@CTReff' + ' metres'),
                    ('Expansion Rate (Past)', '@ExpansionRate{+0.0}' + ' m-2/sec'),
                    ('Rate of Temp. Change', '@CoolingRate{+0.0}' + ' K/15mins'),
                    ('Min. Brightness Temp', '@BTmin{0.0}' + ' K'),
                    ('Average Brightness Temp', '@BTmoy{0.0}' + ' K'),
                    ('Max. Cloud Optical Thickness', '@CTCot'),
                    ('No. of Cloud-Ground Positive Lightning Strokes', '@NbPosLightning')
                ],
                renderers=[renderer])
        figure.add_tools(tool)
        return RenderGroup([renderer, lines, circles, cntr_circles, future_lines, arrows])


class Loader:
    """High-level RDT loader"""
    def __init__(self, pattern):
        self.locator = Locator(pattern)

    def load_date(self, date):
        file_name = self.locator.find_file(date)
        print(file_name)
        if os.path.splitext(file_name)[1] == '.nc':
            return self.load_all_netcdf(file_name)
        elif os.path.splitext(file_name)[1] == '.json':
            return (
                self.load_polygon_json(file_name),
                self.load_tail_lines_json(file_name),
                self.load_tail_points_json(file_name),
                self.load_centre_points_json(file_name)
            )
        else:
            return 'File extension not recognised: ' + file_name

    @staticmethod
    def load_all_netcdf(path):
        """
        Loads polygons from netcdf
        :param path: absolute path and filename
        :return: geojson object for plotting in bokeh
        """

        return getRDT(path, 0, 'All')

    @staticmethod
    def load_polygon_json(path):
        """Load GeoJSON string representation of Polygons from file

        :returns: GeoJSON str
        """

        with open(path) as stream:
            rdt = json.load(stream)

        # Convert units from the netcdf / geojson data file units into something more readable (e.g. could be Kelvin to degrees C or Pa to hPa)
        unitsToRescale = {'Pa' : {'scale':100, 'offset':0, 'Units': 'hPa'} }
        # Get text labels instead of numbers for certain fields
        fieldsToLookup = ['PhaseLife', 'SeverityType', 'SeverityIntensity', 'ConvType', 'CType']

        copy = dict(rdt)
        for i, feature in enumerate(rdt["features"]):
            coordinates = feature['geometry']['coordinates'][0]
            lons, lats = np.asarray(coordinates).T
            x, y = geo.web_mercator(lons, lats)
            c = np.array([x, y]).T.tolist()
            copy["features"][i]['geometry']['coordinates'][0] = c

            for k in feature['properties'].keys():

                # Might be easier to have a units lookup instead of this ...
                deldata, myunits = descale_rdt(k, feature['properties'][k])
                if myunits in unitsToRescale.keys():
                    try:
                        mydict = unitsToRescale.get(myunits)
                        scale, offset, units = mydict.values()
                        conv_data = (feature['properties'][k] / scale) + offset
                        copy['features'][i]['properties'][k] = conv_data
                    except:
                        continue

                if k in fieldsToLookup:
                    try:
                        copy['features'][i]['properties'][k] = fieldValueLUT(k, feature['properties'][k])
                    except:
                        continue
        return json.dumps(copy)

    @staticmethod
    def load_polygon_netcdf(path):
        """
        Loads polygons from netcdf
        :param path: absolute path and filename
        :return: geojson object for plotting in bokeh
        """

        return getRDT(path, 0, 'Polygon')


    @staticmethod
    def load_tail_lines_json(path):
        """Load tail line data from file

        :returns: dict representation suitable for ColumnDataSource
        """
        with open(path) as stream:
            rdt = json.load(stream)

        # Create an empty dictionary
        mydict = get_empty_feature_dict('Tail_Lines')

        # Loop through features
        for i, feature in enumerate(rdt["features"]):
            # Append data from the feature properties to the dictionary
            for k in mydict.keys():
                try:
                    thisdata, units = descale_rdt(k, feature['properties'][k])
                    mydict[k].append(thisdata)
                except:
                    # Do nothing at the moment with the xs and ys
                    if not k in ['xs', 'ys']:
                        mydict[k].append(None)
                    else:
                        continue
            # Takes the trajectory lat/lons, reprojects and
            # puts them in a list within a list
            lons = feature['properties']['LonTrajCellCG']
            lats = feature['properties']['LatTrajCellCG']
            xs, ys = geo.web_mercator(lons, lats)
            mydict['xs'].append(xs)
            mydict['ys'].append(ys)

        return mydict

    @staticmethod
    def load_tail_lines_netcdf(path):
        """
        Loads tail lines from the netcdf file
        :param path: absolute path and filename
        :return: dictionary of data for plotting as a ColumnDataSource in bokeh
        """

        return getRDT(path, 0, 'Tail_Lines')


    @staticmethod
    def load_tail_points_json(path):
        with open(path) as stream:
            rdt = json.load(stream)

        # Create an empty dictionary
        mydict = get_empty_feature_dict('Tail_Points')

        # Loop through features
        for i, feature in enumerate(rdt["features"]):
            # Append data from the feature properties to the dictionary
            # First though, how many points do we have in the trajectory tail?
            npts = len(feature['properties']['LonTrajCellCG'])
            mykeys = [k for k in mydict.keys() if k not in ['x', 'y']]
            for k in mykeys:
                # print(k, type(feature['properties'][k]), sep=':')
                try:
                    if not isinstance(feature['properties'][k], list):
                        datalist = [i for i in itertools.repeat(feature['properties'][k],npts)]
                        thisdata, units = descale_rdt(k, datalist)
                        mydict[k].extend(datalist)
                    else:
                        thisdata, units = descale_rdt(k, feature['properties'][k])
                        mydict[k].extend(thisdata)
                except:
                    datalist = [i for i in itertools.repeat(None, npts)]
                    mydict[k].extend(datalist)
            # Takes the trajectory lat/lons, reprojects and puts them in a list within a list
            lons = feature['properties']['LonTrajCellCG']
            lats = feature['properties']['LatTrajCellCG']
            x, y = geo.web_mercator(lons, lats)
            mydict['x'].extend(x)
            mydict['y'].extend(y)
        return mydict

    @staticmethod
    def load_tail_points_netcdf(path):
        """
        Loads tail points from the netcdf file
        :param path: absolute path and filename
        :return: dictionary of data for plotting as a ColumnDataSource in bokeh
        """

        return getRDT(path, 0, 'Tail_Points')

    @staticmethod
    def load_centre_points_json(path):
        """Holds a centre point, future point and future movement line"""

        with open(path) as stream:
            rdt = json.load(stream)

        # Create an empty dictionary
        mydict = get_empty_feature_dict('Centre_Point')

        # Loop through features
        for i, feature in enumerate(rdt["features"]):
            # Append data from the feature properties to the dictionary
            mykeys = [k for k in mydict.keys() if not (('x' in k) or ('y' in k))]
            for k in mykeys:
                try:
                    thisdata, units = descale_rdt(k, feature['properties'][k])
                    mydict[k].append(thisdata)
                except:
                    mydict[k].append(None)
            # Takes the trajectory lat/lons, reprojects and puts them in a list within a list
            lon = feature['properties']['LonG']
            lat = feature['properties']['LatG']
            x1, y1 = geo.web_mercator(lon, lat)
            mydict['x1'].extend(x1)
            mydict['y1'].extend(y1)

            # Now calculate future point and line
            try:
                speed = float(feature['properties']['MvtSpeed'])
            except ValueError:
                speed = 0
            try:
                direction = float(feature['properties']['MvtDirection'])
            except ValueError:
                direction = 0

            mydict = make_arrow(mydict, lon, lat, speed, direction)

        return mydict

    @staticmethod
    def load_centre_points_netcdf(path):
        """
        Loads tail points from the netcdf file
        :param path: absolute path and filename
        :return: dictionary of data for plotting as a ColumnDataSource in bokeh
        """

        return getRDT(path, 0, 'Centre_Point')


def make_arrow(mydict, lon, lat, speed, direction):

    lon2, lat2 = calc_dst_point(lon, lat, speed, direction)
    x1, y1 = geo.web_mercator(lon, lat)
    x2, y2 = geo.web_mercator(lon2, lat2)
    mydict['x2'].extend(x2)
    mydict['y2'].extend(y2)
    mydict['xs'].append([x1, x2])
    mydict['ys'].append([y1, y2])

    # Now calculate arrow polygon
    x3d, y3d, x4d, y4d = get_arrow_poly(lon2, lat2, speed, direction)
    [x3, x4], [y3, y4] = geo.web_mercator([x3d, x4d], [y3d, y4d])

    mydict['Arrowxs'].append([x2[0], x3, x4])
    mydict['Arrowys'].append([y2[0], y3, y4])

    return mydict

def get_empty_feature_dict(type):

    if type == 'Tail_Lines':
        return dict(
                xs=[], ys=[],
                LonTrajCellCG=[],
                LatTrajCellCG=[],
                NumIdCell=[],
                NumIdBirth=[],
                DTimeTraj=[],
                BTempTraj=[],
                BTminTraj=[],
                BaseAreaTraj=[],
                TopAreaTraj=[],
                CoolingRateTraj=[],
                ExpanRateTraj=[],
                SpeedTraj=[],
                DirTraj=[])

    if type == 'Tail_Points':
        return dict(
                x=[], y=[],
                LonTrajCellCG=[],
                LatTrajCellCG=[],
                NumIdCell=[],
                NumIdBirth=[],
                DTimeTraj=[],
                BTempTraj=[],
                BTminTraj=[],
                BaseAreaTraj=[],
                TopAreaTraj=[],
                CoolingRateTraj=[],
                ExpanRateTraj=[],
                SpeedTraj=[],
                DirTraj=[])

    if type == 'Centre_Point':
        return dict(
                x1=[], y1=[], x2=[], y2=[], xs=[], ys=[],
                Arrowxs=[],
                Arrowys=[],
                LonG=[],
                LatG=[],
                NumIdCell=[],
                NumIdBirth=[],
                MvtSpeed=[],
                MvtDirection=[])
    else:
        return type + ': Not a valid feature type'


def getRDT(path, lev, type):

    '''
    Gets RDT data from the netcdf output of the NWCSAF software
    :param path: Full path and filename of the netcdf file
    :param lev: [0,1] Level number. 0 = bottom of the cloud, 1 = top of the cloud (heights vary between clouds)
    :param type: ['All', 'Centre_Point', 'Polygon', 'Tail_Points', 'Tail_Lines', 'Gridded']
    :return: geojson feature collection object for plotting
    '''

    # Load the netcdf data
    ncds = nc.Dataset(path)

    # Get variable names that have either 'nlevel' or 'recNUM' dimensions
    varlev = [] # 'nlevel' dimension
    varrec = [] # 'recNUM' dimension
    vartraj = [] # 'nbpttraj' Trajectory points for each object
    for ncv in ncds.variables.keys():
        var_dim = ncds.variables[ncv].dimensions
        if 'nlevel' in var_dim:
            varlev.append(ncv)
        if ('recNUM' in var_dim) and (ncds.dimensions['recNUM'].size == ncds.variables[ncv].size):
            varrec.append(ncv)
        if 'nbpttraj' in var_dim:
            vartraj.append(ncv)
    allvars = varlev.copy()
    allvars.extend(varrec)

    # How many cloud levels have we got? It should only be 2 (cloud top and bottom)
    dimsize = len(ma.getdata(ncds.variables[varlev[0]][0,:]))

    # Convert units from the netcdf
    unitsToRescale = {'Pa': 'hPa'} #, 'K': 'degC'}
    # Get text labels instead of numbers for certain fields
    fieldsToLookup = ['PhaseLife', 'SeverityType', 'SeverityIntensity', 'ConvType', 'CType']

    # Check the lev is within the dimsize
    if not lev in np.arange(dimsize):
        return 'Please enter a valid level number (0 or 1)'
    else:
        list_to_return = []

    if type in ['Polygon', 'All']:

        # Create list of features to populate
        features = []
        fieldsToLookup = ['PhaseLife', 'SeverityType', 'SeverityIntensity', 'ConvType', 'CType']
        for i in np.arange(ncds.dimensions['recNUM'].size):

            # do something
            ypolcoords_ll = getDataOnly(ncds.variables['LatContour'][i, lev, :])
            xpolcoords_ll = getDataOnly(ncds.variables['LonContour'][i, lev, :])

            xpolcoords, ypolcoords = geo.web_mercator(xpolcoords_ll, ypolcoords_ll)

            this_poly = Polygon([[(float(coord[0]), float(coord[1])) for coord in zip(xpolcoords, ypolcoords)]])

            # Get the properties for this polygon
            pol_props = {}
            for var in allvars:

                if var in varlev:
                    thisdata = ncds.variables[var][i, lev]
                else:
                    thisdata = ncds.variables[var][i]

                thisdata_scaled = convert_values(thisdata, ncds.variables[var])
                datatype = ncds.variables[var].datatype

                if var in fieldsToLookup:
                    try:
                        thisdata_scaled = fieldValueLUT(var, int(thisdata))
                        datatype = 'string'
                    except:
                        continue

                update_json(pol_props, var, thisdata_scaled, datatype)

            features.append(Feature(geometry=this_poly, properties=pol_props))

        feature_collection = FeatureCollection(features)
        
        list_to_return.append(json.dumps(feature_collection))


    if type in ['Tail_Lines', 'All']:

        # Create an empty dictionary
        mydict_tl = get_empty_feature_dict('Tail_Lines')

        # Loop through features
        for i in np.arange(ncds.dimensions['recNUM'].size):

            # Loop through all items in mydict_tl
            for k in mydict_tl.keys():
                try:
                    thisdata, units = descale_rdt(k, getDataOnly(ncds.variables[k][:]))
                    mydict_tl[k].append(thisdata)
                except:
                    # Do nothing at the moment with the xs and ys
                    if not k in ['xs', 'ys']:
                        mydict_tl[k].append(None)
                    else:
                        continue

            lats = getDataOnly(ncds.variables['LatTrajCellCG'][i, :])
            lons = getDataOnly(ncds.variables['LonTrajCellCG'][i, :])

            xs, ys = geo.web_mercator(lons, lats)
            mydict_tl['xs'].append(xs)
            mydict_tl['ys'].append(ys)

        list_to_return.append(mydict_tl)


    if type in ['Tail_Points', 'All']:

        # Create an empty dictionary
        mydict_tp = get_empty_feature_dict('Tail_Points')

        for i in np.arange(ncds.dimensions['recNUM'].size):

            # do something
            npts = len(getDataOnly(ncds.variables['LonTrajCellCG'][i]))
            mykeys = [k for k in mydict_tp.keys() if k not in ['x', 'y']]
            for k in mykeys:

                thisdata = getDataOnly(ncds.variables[k][i]).tolist()

                try:
                    if len(thisdata) == 1:
                        thisdata = thisdata[0]
                except:
                    pass

                if isinstance(thisdata, list) and (npts == len(thisdata)):
                    datalist, units = descale_rdt(k, thisdata)
                    mydict_tp[k].extend(datalist)
                else:
                    # Repeat the data value npts times
                    datalist = [i for i in itertools.repeat(thisdata, npts)]
                    thisdata, units = descale_rdt(k, datalist)
                    mydict_tp[k].extend(datalist)

                # Some records don't have any data. Mostly happens for ExpanRateTraj
                if len(thisdata) == 0:
                    datalist = [i for i in itertools.repeat('-', npts)]
                    mydict_tp[k].extend(datalist)

                # Some records seem to have missing values
                if len(datalist) != npts:
                    pdb.set_trace()

            lats = getDataOnly(ncds.variables['LatTrajCellCG'][i, :])
            lons = getDataOnly(ncds.variables['LonTrajCellCG'][i, :])
            x, y = geo.web_mercator(lons, lats)
            mydict_tp['x'].extend(x)
            mydict_tp['y'].extend(y)

        list_to_return.append(mydict_tp)

    # Do specific things for each feature type
    if type in ['Centre_Point', 'All']:

        # Create an empty dictionary
        mydict_cp = get_empty_feature_dict('Centre_Point')

        for i in np.arange(ncds.dimensions['recNUM'].size):

            # Get the Point features
            lat = ncds.variables['LatG'][i, lev]
            lon = ncds.variables['LonG'][i, lev]
            x1, y1 = geo.web_mercator(lon, lat)
            mydict_cp['x1'].extend(x1)
            mydict_cp['y1'].extend(y1)

            mykeys = [k for k in mydict_cp.keys() if not (('x' in k) or ('y' in k))]
            for k in mykeys:
                try:
                    if k in varlev:
                        thisdata = ncds.variables[k][i, lev]
                    else:
                        thisdata = ncds.variables[k][i]

                    thisdata, units = descale_rdt(k, thisdata) # May not need to do this
                    mydict_cp[k].append(thisdata)
                except:
                    mydict_cp[k].append(None)

            # Now calculate future point and line
            try:
                speed = float(ncds.variables['MvtSpeed'][i])
            except ValueError:
                speed = 0
            try:
                direction = float(ncds.variables['MvtDirection'][i])
            except ValueError:
                direction = 0

            mydict_cp = make_arrow(mydict_cp, lon, lat, speed, direction)

        list_to_return.append(mydict_cp)


    if len(list_to_return) == 1:
        return list_to_return[0]
    elif len(list_to_return) > 1:
        return tuple(list_to_return)
    else:
        return 'Nothing to return'


def getDataOnly(array1d):
    '''
    Removes redundant no data slots in a 1D array
    Note that sometimes data is missing within the array, so in the majority of cases, the array just needs to be shortened, but in a small number of cases, the 'within array' no data needs to be replaced
    :param array1d:
    :return: Simple array of numbers
    '''

    if np.any(ma.getmask(array1d)):
        mymask = np.invert(ma.getmask(array1d))
        outarray = ma.getdata(array1d)[mymask]
    else:
        outarray = ma.getdata(array1d)

    return outarray

def update_json(props, varname, data, datatype):
    """
    Adds an extra field to the properties of a feature in a geojson file. Provides a consistent way to handle different data types
    :param props: dictionary of properties
    :param varname: string variable name
    :param data: the data to add
    :param datatype: datatype
    :return: updated props
    """

    if isinstance(data, ma.MaskedArray) and (data.shape == ()) and ('int' in str(datatype)):
        props.update({varname: np.int(ma.getdata(data))})

    elif isinstance(data, ma.MaskedArray) and (data.shape == ()) and ('float' in str(datatype)):
        props.update({varname: np.float(ma.getdata(data))})

    elif isinstance(data, np.float32) or isinstance(data, np.float):
        props.update({varname: float(data)})

    elif isinstance(data, np.int) or isinstance(data, np.uint16):
        props.update({varname: int(data)})

    elif str(datatype) == 'string':
        props.update({varname: str(data)})

    else:
        return



def calc_dst_point(x1d, y1d, speed, angle):
    """Calculate destination point

    Estimates positions in longitude/latitude space from speed and
    angle on the surface of a sphere, in this case Earth.

    :param x1d: longitude
    :param y1d: latitude
    """
    # NB: X and Y need to be lat/lons

    # Distance travelled (m) = speed (m/s) * 60 seconds * 60 minutes
    # NB: 60 mins may change depending on the time frequency of the display (currently 1 hour)
    d = (speed * 60 * 60)

    # Radius of the earth (m)
    R = 6378137

    x1 = math.radians(x1d)
    y1 = math.radians(y1d)

    # Convert degrees to radians
    direction = math.radians(angle)

    y2 = math.asin(math.sin(y1) * math.cos(d / R) +
                   math.cos(y1) * math.sin(d / R) * math.cos(direction))

    x2 = x1 + math.atan2(math.sin(direction) * math.sin(d / R) * math.cos(y1),
                         math.cos(d / R) - math.sin(y1) * math.sin(y2))

    x2d = math.degrees(x2)
    y2d = math.degrees(y2)
    return x2d, y2d


def get_arrow_poly(x2,y2, speed, direction):
    """Draw a polygon representing an arrow in geographical coordinates

    .. note:: The arrows are scaled in longitude/latitude space not
              in screen coordinates

    :param x2: longitude of point
    :param y2: latitude of point
    :param speed: scalar velocity in m/s
    :param direction: angle in degrees relative to north
    """
    timestep = 60 # See above function re: 60 mins
    mvt_line_len = speed * 60 * timestep
    mvt_line_dir = direction
    arrow_angl = 20
    arrow_linefrac = 1./5

    # First point
    pt1_dir = (mvt_line_dir - 180) % 360 - arrow_angl
    pt1_len = math.sqrt(3. * math.pow( mvt_line_len * arrow_linefrac, 2 ) / 2) # Metres
    # Convert len back to speed for the function
    pt1_speed = pt1_len / (timestep * 60)
    # Calculate x3, y3
    x3, y3 = calc_dst_point(x2, y2, pt1_speed, pt1_dir)

    # Second point
    pt2_dir = (mvt_line_dir - 180) % 360 + arrow_angl
    pt2_len = math.sqrt(3.* math.pow( mvt_line_len * arrow_linefrac, 2 ) / 2) # Metres
    # Convert len back to speed for the function
    pt2_speed = pt2_len / (timestep * 60)
    # Calculate x3, y3
    x4, y4 = calc_dst_point(x2, y2, pt2_speed, pt2_dir)
    return x3, y3, x4, y4


def convert_values(conv_value, conv_var):
    """ Specific conversions for GeoJSON storage efficiency """

    DP_ACCURACY = 0

    if conv_var.name == 'ExpansionRate':
        try:
            return int(round(conv_value * 360000, DP_ACCURACY))  # to %/hr
        except TypeError:
            return conv_value

    elif conv_var.name == 'CoolingRate':
        try:
            return int(round(conv_value * 3600, DP_ACCURACY))  # to K/hr
        except TypeError:
            return conv_value

    elif conv_var.name == 'Surface':
        try:
            return int(round(conv_value / 1e6, DP_ACCURACY))  # to km2
        except TypeError:
            return conv_value

    elif conv_var.name == 'CTPressure':
        myu = cf_units.Unit('Pa')
        try:
            return int(round(myu.convert(conv_value, 'hPa'), DP_ACCURACY))
        except TypeError:
            return conv_value

    elif conv_var.name == 'CRainRate':
        try:
            return int(round(conv_value, DP_ACCURACY)) # Round and make it an integer
        except TypeError:
            return conv_value

    elif conv_var.name == 'CTPressRate':
        myu = cf_units.Unit('Pa s-1')
        try:
            return int(round(myu.convert(conv_value, 'hPa h-1'), DP_ACCURACY)) # Pa/s to hPa/hour
        except TypeError:
            return conv_value

    elif conv_var.name in ['BTemp', 'BTmin', 'BTmoy']:
        myu = cf_units.Unit('K')
        try:
            return round(myu.convert(conv_value, 'degC'), 1)
        except TypeError:
            return conv_value

    else:
        return conv_value


def descale_rdt(fn, data):
    # Converts units according to netcdf files definition
    rdtUnitsLUT = {
        'DecTime': {'scale': 1, 'offset': 0, 'Units': 's'},
        'LeadTime': {'scale': 1, 'offset': 0, 'Units': 's'},
        'Duration': {'scale': 1, 'offset': 0, 'Units': 's'},
        'MvtSpeed': {'scale': 0.001, 'offset': 0, 'Units': 'm s-1'},
        'MvtDirection': {'scale': 1, 'offset': 0, 'Units': 'degree'},
        'DtTimeRate': {'scale': 1, 'offset': 0, 'Units': 's'},
        'ExpansionRate': {'scale': 2e-07, 'offset': -0.005, 'Units': 's-1'},
        'CoolingRate': {'scale': 2e-06, 'offset': -0.05, 'Units': 'K s-1'},
        'LightningRate': {'scale': 1e-04, 'offset': -2.0, 'Units': 's-1'},
        'CTPressRate': {'scale': 0.001, 'offset': -25.0, 'Units': 'Pa s-1'},
        'BTemp': {'scale': 0.01, 'offset': 130.0, 'Units': 'K'},
        'BTmoy': {'scale': 0.01, 'offset': 130.0, 'Units': 'K'},
        'BTmin': {'scale': 0.01, 'offset': 130.0, 'Units': 'K'},
        'Surface': {'scale': 5000000.0, 'offset': 0, 'Units': 'm2'},
        'EllipseGaxe': {'scale': 20.0, 'offset': 0, 'Units': 'm'},
        'EllipsePaxe': {'scale': 20.0, 'offset': 0, 'Units': 'm'},
        'EllipseAngle': {'scale': 1, 'offset': 0, 'Units': 'degrees_north'},
        'DtLightning': {'scale': 1, 'offset': 0, 'Units': 's'},
        'CTPressure': {'scale': 10.0, 'offset': 0, 'Units': 'Pa'},
        'CTCot': {'scale': 0.01, 'offset': 0, 'Units': '1'},
        'CTReff': {'scale': 1e-08, 'offset': 0, 'Units': 'm'},
        'CTCwp': {'scale': 0.001, 'offset': 0, 'Units': 'kg m-2'},
        'CRainRate': {'scale': 0.1, 'offset': 0, 'Units': 'mm/h'},
        'BTempSlice': {'scale': 0.01, 'offset': 130.0, 'Units': 'K'},
        'SurfaceSlice': {'scale': 5000000.0, 'offset': 0, 'Units': 'm2'},
        'DTimeTraj': {'scale': 1, 'offset': 0, 'Units': 's'},
        'BTempTraj': {'scale': 0.01, 'offset': 130.0, 'Units': 'K'},
        'BTminTraj': {'scale': 0.01, 'offset': 130.0, 'Units': 'K'},
        'BaseAreaTraj': {'scale': 5000000.0, 'offset': 0, 'Units': 'm2'},
        'TopAreaTraj': {'scale': 5000000.0, 'offset': 0, 'Units': 'm2'},
        'CoolingRateTraj': {'scale': 2e-06, 'offset': -0.05, 'Units': 'K s-1'},
        'ExpanRateTraj': {'scale': 2e-07, 'offset': -0.005, 'Units': 's-1'},
        'SpeedTraj': {'scale': 0.001, 'offset': 0, 'Units': 'm s-1'},
        'DirTraj': {'scale': 1, 'offset': 0, 'Units': 'degree'}
    }

    try:
        dict = rdtUnitsLUT.get(fn, {'scale': 1, 'offset': 0, 'units': '-'})
        scale, offset, units = dict.values()
        conv_data = ( data / scale ) + offset
        return(conv_data, units)
    except:
        return(data, '-')


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
            1: "Discrimination statistical scheme",
            2: "Electric",
            3: "Overshoot",
            4: "Convective rain rate",
            5: "Tropical"
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
            0: "No activity",
            1: "Turbulence",
            2: "Lightning",
            3: "Icing",
            4: "High altitude icing",
            5: "Hail",
            6: "Heavy rainfall",
            7: "not defined"
        },
        'SeverityIntensity' : {
            0: "not defined",
            1: "Low",
            2: "Moderate",
            3: "High",
            4: "Very high"
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
            0: "not defined",
            1: "Low risk",
            2: "Moderate risk",
            3: "High risk-free",
            4: "Very high risk"
        }
    }
    try:
        return rdtLUT.get(fn)[uid]
    except:
        return "-"


class Locator:
    def __init__(self, pattern):
        self.pattern = pattern

    def valid_times(self):
        """Parse file names to an array of dates"""
        paths = self.find(self.pattern)
        parsed_times = [self.parse_date(p) for p in paths]
        return sorted(set(t for t in parsed_times if t is not None))

    def find_file(self, valid_date):
        paths = np.array(self.paths)  # Note: timeout cache in use)
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
    @forest.util.timeout_cache(dt.timedelta(minutes=10))
    def find(pattern):
        return sorted(glob.glob(pattern))

    def dates(self, paths):
        return np.array([
            self.parse_date(p) for p in paths],
            dtype='datetime64[s]')

    @staticmethod
    def parse_date(path):
        if os.path.splitext(path)[1] == '.json':
            groups = re.search(r"[0-9]{12}", os.path.basename(path))
            if groups is not None:
                return dt.datetime.strptime(groups[0], "%Y%m%d%H%M")
        elif os.path.splitext(path)[1] == '.nc':
            groups = re.search(r"[0-9]{8}T[0-9]{6}", os.path.basename(path))
            if groups is not None:
                return dt.datetime.strptime(groups[0], "%Y%m%dT%H%M%S")
        else:
            return 'Unable to parse datetime from filename'
