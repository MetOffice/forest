

class Navigator:
    """High-level Navigator

    :param navigators: dict of sub-navigators distinguished by pattern
    """
    def __init__(self, _navigators):
        self._navigators = _navigators

    def variables(self, pattern):
        navigator = self._navigators[pattern]
        return navigator.variables(pattern)

    def initial_times(self, pattern, variable=None):
        navigator = self._navigators[pattern]
        return navigator.initial_times(pattern, variable=variable)

    def valid_times(self, pattern, variable, initial_time):
        navigator = self._navigators[pattern]
        return navigator.valid_times(pattern, variable, initial_time)

    def pressures(self, pattern, variable, initial_time):
        navigator = self._navigators[pattern]
        return navigator.pressures(pattern, variable, initial_time)
