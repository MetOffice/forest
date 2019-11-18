"""Indirect access to State properties"""


class Selector:
    """Access data stored in state"""
    _props = [
        "pressure",
        "pressures",
        "valid_time",
        "initial_time",
        "variable"
    ]
    def __init__(self, state):
        self.state = state

    def defined(self, attr):
        """Determine if property defined"""
        if isinstance(self.state, tuple):
            return getattr(self.state, attr, None) is not None
        return attr in self.state

    def __getattr__(self, attr):
        if attr in self._props:
            return self._get(attr)
        return self.__dict__.get(attr)

    def _get(self, attr):
        if isinstance(self.state, tuple):
            return getattr(self.state, attr, None)
        return self.state.get(attr, None)
