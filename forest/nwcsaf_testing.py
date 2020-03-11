# Some notes on what I've done with the NWCSAF output

import xarray
import iris
import datashader
from datashader import transfer_functions as tf

cubes = iris.load('/scratch/hadhy/NWCSAF/S_NWC_PC_MSG4_eastafrica-VISIR_20200211T180000Z.nc')
c = cubes[0]

lons = c.coords('longitude')[0].points
lats = c.coords('latitude')[0].points
e = xarray.DataArray(c.data, coords=[('y', lats[:, 0]), ('x', lons[0, :])], name='Z')
canvas = datashader.Canvas(plot_height=4, plot_width=4)
myarray = tf.Image(tf.shade(canvas.quadmesh(e, x='x', y='y')))

xcoords = myarray.coords['x'].data
ycoords = myarray.coords['y'].data

data2plot = myarray.data

