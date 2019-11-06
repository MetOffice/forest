import io
import json
import os
from datetime import datetime
from collections import namedtuple

import requests
import xarray as xr
import numpy as np

from forest import geo, gridded_forecast

# replace this with a live collection (or code to auto find collections)
# see https://mdda.hub.metoffice.cloud/documentation
URL = 'https://mdda.hub.metoffice.cloud/v1/collections/PT1HTE1fMjAxOS0xMC0yOVQwNi4wMC4wMFpfSVNCTA=='
APIKEY = os.environ['MDDA_KEY']

VALID_TIME = datetime(2019, 10, 24, 0, 0, 0)

EXAMPLE_RESPONSE = {
  "queryStringParameters": {
    "outputFormats": [
      "netCDF4",
      "GRIB2"
    ],
    "parameters": {
      "relative-humidity": {
        "description": {
          "en": "Relative-humidity"
        },
        "observedProperty": {
          "id": "http://codes.wmo.int/grib2/codeflag/4.2/_0-1-1",
          "label": {
            "en": "Relative-humidity"
          }
        },
        "type": "Parameter",
        "unit": {
          "label": {
            "en": "TODO (e.g. kelvin)"
          },
          "symbol": {
            "type": "http://www.opengis.net/def/uom/UCUM/",
            "value": 1
          }
        },
        "validParameters": {
          "time": {
            "items": [
              "2019-06-25T00:00:00Z",
              "2019-06-25T01:00:00Z",
              "2019-06-25T02:00:00Z"
            ],
            "name": "Time",
            "units": "ISO8601"
          },
          "z": {
            "items": [
              2,
              5,
              10
            ],
            "name": "altitude",
            "units": "m"
          }
        }
      }
    },
    "ranges": {
      "xRange": {
        "label": "Lon",
        "lowerBound": "string",
        "upperBound": "string",
        "uomLabel": "deg"
      },
      "yRange": {
        "label": "Lat",
        "lowerBound": "string",
        "upperBound": "string",
        "uomLabel": "deg"
      }
    }
  }
}

class MddaResponse:
    def __init__(self, mdda_response):
        self.content = mdda_response

    @property
    def reference_times(self):
        return [VALID_TIME]

    @property
    def parameters(self):
        return list(self.content['queryStringParameters']['parameters'].keys())

    def parameter_times(self, parameter_id):
        parameter_metadata = self.content['queryStringParameters']['parameters'][parameter_id]
        times = parameter_metadata['validParameters']['time']['items']
        parsed_times = list(map(
            lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ"),
            times))
        return parsed_times

    def parameter_levels(self, parameter_id):
        parameter_metadata = self.content['queryStringParameters']['parameters'][parameter_id]
        levels = parameter_metadata['validParameters']['z']['items']
        parsed_levels = list(map(float, levels))
        return parsed_levels

def _load_from_mdda(url):
    if url != None:
        print(url)
        response = requests.get(
            url, headers={'x-api-key': APIKEY})
        print(response)
        collection = json.loads(response.content)
    else:
        collection = EXAMPLE_RESPONSE
    return MddaResponse(collection)

def _nc_from_mdda(parameter, level, t):
    print('formatting time')
    t = gridded_forecast._to_datetime(t)
    print(t)
    time_str = t.strftime('%Y-%m-%dT%H:%M:%SZ')
    print('getting data')
    print(level)
    data = requests.get(
        f'{URL}/cube',
        params = {
            'parameters': parameter,
            'outputFormat': 'netCDF4',
            'bbox': '-180,-90,180,90',
            'time': time_str,
            'z': level},
        headers = {
            'x-api-key': APIKEY
            })
    print(data.url)
    print(data.headers)
    if data.headers['Content-Type'] == 'text/html':
        print(data.content)
    da = xr.open_dataset(io.BytesIO(data.content))[parameter]
    del da.attrs['standard_name'] # mdda returns invalid standard names
    cube = da.to_iris()
    cube = cube[0,0,0] # drop ref time, level, and valid
    return cube

class MddaLoader:
    def __init__(self, url=URL):
        print('mdda loader')
        self._label = 'dummy data'
        self.metadata = _load_from_mdda(url)

    def image(self, state):
        print('mdda image')
        variable = state.variable
        init_time = state.initial_time
        valid_time = state.valid_time
        pressure = state.pressure
        print(state)
        cube = _nc_from_mdda(variable, pressure, valid_time)
        print('hey I am cube')
        print(cube)
        if cube is None or state.initial_time is None:
            data = gridded_forecast.empty_image()
            print('returning empty data:')
        else:
            print(cube)
            print(type(cube))
            data = geo.stretch_image(cube.coord('longitude').points,
                                        cube.coord('latitude').points, cube.data)
            img = data['image'][0]
            masked_img = np.ma.masked_array(img, mask = np.isnan(img))
            data['image'] = [masked_img]
            data.update(gridded_forecast.coordinates(state.valid_time, state.initial_time,
                                    state.pressures, state.pressure))
            data.update({
                'name': [self._label],
                'units': [str(cube.units)]
            })
        print('returning data:')
        print(data)
        return data


class MddaNavigator:
    def __init__(self, url=URL):
        print('navigator')
        print(url)
        self.metadata = _load_from_mdda(url)

    def variables(self, *args, **kwargs):
        p = self.metadata.parameters
        print(p)
        return p

    def initial_times(self, *args, **kwargs):
        print('init times')
        print(self.metadata.reference_times)
        return self.metadata.reference_times

    def valid_times(self, *args, **kwargs):
        # not all variables are necessarily available on all times
        params = self.metadata.parameters
        times = [self.metadata.parameter_times(p) for p in params]
        longest_t = sorted(times, reverse = True, key = lambda arr: len(arr))[0]
        print('valid times')
        print(longest_t)
        return longest_t

    def pressures(self, *args, **kwargs):
        print('pressures')
        # not all variables are necessarily available on all levels
        # data is not necessarily on pressure levels
        params = self.metadata.parameters
        levels = [self.metadata.parameter_levels(p) for p in params]
        longest_l = sorted(levels, reverse = True, key = lambda arr: len(arr))[0]
        return longest_l

if __name__ == '__main__':
    
    print('with dummy responses')
    navigator = MddaNavigator(url=None)
    print(navigator.initial_times())
    print(navigator.valid_times())
    print(navigator.pressures())
    print(navigator.variables())

    print('with live data...')
    navigator = MddaNavigator()
    print(navigator.initial_times())
    print(navigator.valid_times())
    print(navigator.pressures())
    print(navigator.variables())
    
    State = namedtuple('State', field_names=['variable', 'initial_time', 'valid_time', 'pressures', 'pressure'])
    state = State(
        'temperature',
        navigator.initial_times()[0],
        navigator.valid_times()[0],
        navigator.pressures(),
        navigator.pressures()[0])

    loader = MddaLoader()
    print(loader.image(state))
