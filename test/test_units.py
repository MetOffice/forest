import netCDF4
import os
from forest.drivers.unified_model import Loader

# example where there are units in the file
def test_readunits():
    filename = 'dummyfile.nc'
    parameter = 'mslp'
    dataset = netCDF4.Dataset(filename,'w')
    v = dataset.createVariable('mslp','f',())
    v.units = 'hPa'
    dataset.close()
    result = Loader.read_units(filename,parameter)
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
    result = Loader.read_units(filename,parameter)
    expect = ''
    assert result == expect
    os.remove(filename)
