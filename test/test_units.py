import netCDF4
import os


def read_units(filename,parameter):
    dataset = netCDF4.Dataset(filename)
    units = dataset.variables[parameter].units
    dataset.close()
    return units


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

def test_read_no_units():
    filename = 'dummybadfile.nc'
    parameter = 'nonsense'
    dataset = netCDF4.Dataset(filename,'w')
    NNN = dataset.createVariable('nonsense','f',())
    NNN.units = ''
    dataset.close()
    result = read_units(filename,parameter)
    expect = ''
    assert result == expect
    os.remove(filename)
