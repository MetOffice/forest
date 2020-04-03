import pytest
import netCDF4
from forest.drivers.unified_model import Loader


@pytest.mark.parametrize("parameter,units,expect", [
    pytest.param('mslp', 'hPa', 'hPa', id="read units"),
    pytest.param('nonsense', None, '', id="no units")
])
def test_readunits(tmpdir, parameter, units, expect):
    filename = str(tmpdir / "file.nc")
    with netCDF4.Dataset(filename,'w') as dataset:
        # Longitude
        dataset.createDimension("longitude", 2)
        var = dataset.createVariable("longitude", "f", ("longitude",))
        var[:] = [0, 1]

        # Latitude
        dataset.createDimension("latitude", 2)
        var = dataset.createVariable("latitude", "f", ("latitude",))
        var[:] = [0, 1]

        # Variable
        v = dataset.createVariable(parameter,'f',("longitude", "latitude"))
        if units is not None:
            v.units = units

    data = Loader.load_image(filename, parameter, ())
    assert data["units"][0] == expect
