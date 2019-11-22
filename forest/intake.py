'''
Module for loading data from a generic intake catalogue.
TODO: describe the expected structure of the intake catalogue.
TODO: handle gridded netcdf and tabular CSV data.
'''


try:
    import iris
except ModuleNotFoundError:
    iris = None
    # ReadTheDocs can't import iris

try:
    import intake
except ModuleNotFoundError:
    intake = None

from forest import geo, gridded_forecast



class IntakeLoader:
    """
    Loader class for a generic intake catalogue.
    """
    def __init__(self, pattern):


    def image(self, state):
        """
        Main image loading function. This function will actually realise the
        data,
        """
        data = gridded_forecast.empty_image()
        data.update({
            'name': ['intake_data'],
            'units': ['No units'],
        })

        return data


class Intakeavigator:
    """
    Navigator class for CMIP6 intake dataset.
    """

    def __init__(self):
        raise NotImplementedError()

    def variables(self, pattern):
        raise NotImplementedError()

    def initial_times(self, pattern, variable=None):
        raise NotImplementedError()

    def valid_times(self, pattern, variable, initial_time):
        raise NotImplementedError()

    def pressures(self, pattern, variable, initial_time):
        raise NotImplementedError()
