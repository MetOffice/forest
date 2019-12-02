from forest import geo
import pytest

@pytest.mark.parametrize(
'lon,lat,data,expected',
    [([],[],[],{'x':[],'y':[],'dh':[],'dw':[],'image':[]}),
])
def test_ds_pipeline(lon, lat, data, expected):
    output = geo.bokeh_image(lon, lat, data)
    assert output == expected
