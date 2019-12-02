from forest import geo
import pytest

@pytest.mark.parametrize(
'x,y,data,expected',
    [([],[],[],{'x':[],'y':[],'dh':[],'dw':[],'image':[]}),
])
def test_ds_pipeline(x, y, data, expected):
    output = geo.bokeh_image(x, y, data)
    assert output == expected
