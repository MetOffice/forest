

class Navigator:
    """High-level Navigator

    :param navigators: dict of sub-navigators distinguished by pattern
    """
    def __init__(self, _navigators):
        # TODO: It'd be good to switch the identification of navigators from
        # using the `pattern` to using the `label`. In general, not every
        # group would have a `pattern`.
        # e.g.
        # self._navigators = {label: navigator for label, navigator in ...}
        self._navigators = _navigators

    def __call__(self, store, action):
        """Pass through to appropriate sub-navigator"""
        pattern = store.state.get("pattern")
        if pattern is not None:
            try:
                yield from self._navigators[pattern](store, action)
            except TypeError:
                # Sub-navigator not middleware pass on action
                yield action
        else:
            # Pattern not yet set
            yield action

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
