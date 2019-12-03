import datashader
import numpy
import pytest
import xarray

from forest import geo

def f(x,y):
    return numpy.cos((x**2+y**2)**2)

def sample(fn, n, range):
    xs = ys = numpy.linspace(range_[0], range_[1], n)
    x,y = numpy.meshgrid(xs, ys)
    z   = fn(x,y)
    return xarray.DataArray(z, coords=[('y',ys), ('x',xs)])


@pytest.mark.parametrize(
'x,y,data,expected',
    [([],[],[],{'x':[],'y':[],'dh':[],'dw':[],'image':[]}),
])
def test_ds_pipeline(x, y, data, expected):
    output = geo.bokeh_image(x, y, data)
    assert output == expected


def test_raster_identity():
    test_vals =  xarray.DataArray([[5,2],[3,4]],coords=[('x',[0.0,1.0]),('y',[0.0,1.0])],name='Z')
    test_canvas = datashader.Canvas(plot_height=2, plot_width=2, x_range=(0.0, 1.0),
                                    y_range=(0.0, 1.0))
    output_arr = test_canvas.quadmesh(test_vals)
    assert output_arr.shape == (2,2)
    numpy.testing.assert_array_equal(output_arr, test_vals)

def test_raster_data():
    test_vals =  xarray.DataArray([[5,2],[3,4]],coords=[('x',[0.1,0.9]),('y',[0.1,0.9])])
    test_canvas = datashader.Canvas(plot_height=5, plot_width=5, x_range=(0.0, 1.0),
                                    y_range=(0.0, 1.0))
    output_arr = test_canvas.raster(test_vals, interpolate='nearest')
    assert output_arr.shape == (5,5)
    ref_arr =  xarray.DataArray([[5,5,5,2,2],[5,5,5,2,2],[5,5,5,2,2],[3,3,3,4,4],[3,3,3,4,4]],coords=[('x',[0.1,0.3, 0.5, 0.7, 0.9]),('y',[0.1,0.3, 0.5, 0.7, 0.9])])
    numpy.testing.assert_array_equal(output_arr, ref_arr)


def test_datashder_stretch():
    gx = numpy.array([0.25, 0.75])
    gy = numpy.array([0.25, 0.75])
    x_range = numpy.array([0.0, 1.0])
    y_range = numpy.array([0.0, 1.0])
    values = numpy.array([[0.0, 1.0], [2.0, 3.0]])
    output_image = geo.datashader_stretch(values, gx, gy, x_range, y_range)
    reference_image = xarray.DataArray(values,
                                       coords=[('x', [0.25, 0.75]), ('y', [0.25, 0.75])])
    xarray.testing.assert_equal(output_image, reference_image)
