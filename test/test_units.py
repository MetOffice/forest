import netCDF4
import os


def read_units(filename,parameter):
    dataset = netCDF4.Dataset(filename)
    veep = dataset.variables[parameter]
    # read the units and assign a blank value if there aren't any:
    units = getattr(veep, 'units', '')
    dataset.close()
    return units

# example where there are units in the file
def test_readunits():
    filename = 'dummyfile.nc'
    parameter = 'mslp'
    dataset = netCDF4.Dataset(filename,'w')
    v = dataset.createVariable('mslp','f',())
    v.units = 'hPa'
    dataset.close()
    result = read_units(filename,parameter)
    expect = 'hPa'
    assert result == expect
    os.remove(filename)

# example where there are no units in the file
def test_read_no_units():
    filename = 'dummybadfile.nc'
    parameter = 'nonsense'
    dataset = netCDF4.Dataset(filename,'w')
    NNN = dataset.createVariable('nonsense','f',())
    dataset.close()
    result = read_units(filename,parameter)
    expect = ''
    assert result == expect
    os.remove(filename)
