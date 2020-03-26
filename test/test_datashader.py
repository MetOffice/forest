import numpy
import pytest
import xarray

from forest import geo

try:
    import datashader
    import xarray
    from bokeh.core.json_encoder import serialize_json
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

@pytest.mark.skipif(not libs_available,
                    reason='datashader and xarray are optional')
def test_datashader_image_nan():
    '''Test the datashader image stretch function works correctly 
    if the inputs 2D arraysi containing NaNs in both coordinates and 
    values.
    
    Tests the modification made in commit 05962ec'''
    x = numpy.array(
      [[ numpy.nan,  numpy.nan,  numpy.nan, numpy.nan, numpy.nan, numpy.nan],
       [ numpy.nan,  numpy.nan,  numpy.nan, numpy.nan, numpy.nan, numpy.nan],
       [ numpy.nan,  numpy.nan,  numpy.nan, numpy.nan, numpy.nan, numpy.nan],
       [-32.12598 , -32.07721 , -32.02848 , numpy.nan, numpy.nan, numpy.nan],
       [-32.15292 , -32.104084, -32.055298, numpy.nan, numpy.nan, numpy.nan],
       [-32.17996 , -32.13106 , -32.08221 , numpy.nan, numpy.nan, numpy.nan]])
    y = numpy.array(
      [[ numpy.nan,  numpy.nan,  numpy.nan, numpy.nan, numpy.nan, numpy.nan],
       [ numpy.nan,  numpy.nan,  numpy.nan, numpy.nan, numpy.nan, numpy.nan],
       [ numpy.nan,  numpy.nan,  numpy.nan, numpy.nan, numpy.nan, numpy.nan],
       [-40.60027 , -40.596905, -40.593544, numpy.nan, numpy.nan, numpy.nan],
       [-40.643642, -40.640266, -40.636898, numpy.nan, numpy.nan, numpy.nan],
       [-40.68706 , -40.683674, -40.680298, numpy.nan, numpy.nan, numpy.nan]])
    z = numpy.array(
      [[ numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan],
       [ numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan],
       [ 276.23   , 273.04   , 270.75   , numpy.nan, numpy.nan, numpy.nan],
       [ 277.12   , 273.55   , 270.82   , numpy.nan, numpy.nan, numpy.nan],
       [ numpy.nan, 273.24   , 270.16998, numpy.nan, numpy.nan, numpy.nan],
       [ numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan]])

    result = geo.stretch_image(x, y, z)

    #this will fail on a ValueError if the NaNs above are handled improperly
    serialize_json(result)

    #Should be returning NumPy masked arrays
    assert numpy.ma.is_masked(result['image'][0])
