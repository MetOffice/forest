import numpy
import pytest
import xarray

from forest import geo

try:
    import datashader
    import xarray
    libs_available = True
except ModuleNotFoundError:
    libs_available = False


@pytest.mark.skipif(not libs_available,
                    reason='datashader and xarray are optional')
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


def test_custom_stretch():
    x = numpy.array([0, 1])
    y = numpy.array([0, 1, 2])
    z = numpy.array([[0, 1], [2, 3], [4, 5]])
    numpy.testing.assert_array_equal(geo.custom_stretch(z, x, y), z)


@pytest.mark.skipif(not libs_available,
                    reason='datashader and xarray are optional')
def test_datashader_stretch_image():
    x = numpy.array([0, 1])
    y = numpy.array([0, 1, 2])
    z = numpy.array([[0, 1], [2, 3], [4, 5]])
    x_range = (-0.5, 1.5)
    y_range = (-0.5, 2.5)
    result = geo.datashader_stretch(z, x, y, x_range, y_range)
    expect = z
    numpy.testing.assert_array_equal(result, expect)
