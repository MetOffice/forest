from db.exceptions import SearchFail


class Navigator(object):
    def __init__(self, paths):
        self.paths = paths

    def variables(self, pattern):
        return ['air_temperature']

    def initial_times(self, pattern, variable=None):
        return ['2019-01-01 00:00:00']

    def valid_times(self, pattern, variable, initial_time):
        return ['2019-01-01 12:00:00']

    def pressures(self, pattern, variable, initial_time):
        return [750.]


class Locator(object):
    def locate(
            self,
            pattern,
            variable,
            initial_time,
            valid_time,
            pressure=None,
            tolerance=0.001):
        raise SearchFail
