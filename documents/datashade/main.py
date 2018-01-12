import xarray as xr
import datashader as ds
#import geoviews as gv
import holoviews as hv
import geoviews as gv
 
import geoviews.feature as gf

from holoviews.operation.datashader import regrid, shade
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from holoviews.operation.datashader import aggregate, shade, datashade, dynspread, regrid
from bokeh.layouts import widgetbox
from bokeh.models.widgets import RadioGroup
from bokeh.plotting import curdoc
from bokeh.layouts import column, row
from holoviews.streams import Stream, param
import urllib.request
import os 


try:
    get_ipython
    is_notbook = True
except:
    is_notebook = False

if is_notebook:
    hv.notebook_extension('bokeh', width=80)
else:
    hv.extension('bokeh')
    renderer = hv.renderer('bokeh')
    renderer = renderer.instance(mode='server')
    
# renderer = renderer.instance(mode='server')

use_pysssix = True
sea_data = True

if sea_data:
    bucket = 'stephen-sea-public'
    key = 'model_data/SEA_km4p4_ra1t_20171105T0000Z.nc'
    params = ['air_temperature','y_wind','x_wind']
else:
    bucket = 'mogreps-uk'
    key = 'prods_op_mogreps-uk_20130101_03_00_003.nc'
    params = ['air_temperature','y_wind','x_wind']



if use_pysssix:
    file = os.path.join('/s3/', bucket, key) 
else:
    file = os.path.join('.', os.path.basename(key))

if not os.path.exists(file):
    url = "https://s3.eu-west-2.amazonaws.com/" + bucket + '/' + key
    urllib.request.urlretrieve(url, file)

      


dataset = xr.open_dataset(file, chunks={'time': 1})

Param = Stream.define('Param', p=params[0] ,time=0)

def load_image(time, p):
    print("time %s, p %s" % (time, p))
    data = dataset[p][3]
    if not sea_data:
        data = data[0]
    img = gv.operation.project_image(gv.Image(data))
    
    return img


# image_stack = hv.DynamicMap(load_image,  kdims=['time', 'p'], streams=[Param()])
#plot = regrid(image_stack)# * gf.coastline(plot=dict(scale='10m')) #.redim.range(time=(0,3)) 
gf.coastline.set_param("extents", (90, -18, 154, 30))
coastline = gf.coastline(plot=dict(scale='50m'))
#regridded_map = regrid(load_image(0, params[0]))
data_map = hv.DynamicMap(load_image, streams=[Param()])
regridded_map = regrid(data_map)

print("coastline: %s, regridded_map %s" %(coastline, regridded_map))
plot =  regridded_map * coastline
print("plot: %s" % (plot))
print(plot.id);




def select_param(param=params[0]):
    data_map.event(p=param)

def select_param_handler(index):
    select_param(params[index])

radio_group = RadioGroup(
        labels=params, active=0)

radio_group.on_click(select_param_handler)

if not is_notebook:
    hvplot = renderer.get_plot(plot)
    curdoc().add_root(column(hvplot.state,radio_group))
plot