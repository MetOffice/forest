import numpy
import pytest
import xarray

from forest import geo

def test_datashder_stretch():
    gx = numpy.array([0.25, 0.75])
    gy = numpy.array([0.25, 0.75])
    x_range = numpy.array([0.0, 1.0])
    y_range = numpy.array([0.0, 1.0])
    values = numpy.array([[0.0, 1.0], [2.0, 3.0]])
    output_image = geo.datashader_stretch(values, gx, gy, x_range, y_range)
    reference_image = xarray.DataArray(values,
                                       coords=[('x', [0.25, 0.75]),
                                               ('y', [0.25, 0.75])])
    assert isinstance(output_image, numpy.ndarray)
    numpy.testing.assert_array_equal(output_image, reference_image)
