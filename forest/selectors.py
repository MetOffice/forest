"""Indirect access to State properties"""


class Selector:
    """Access data stored in state"""
    def __init__(self, state):
        self.state = state

    def defined(self, attr):
        """Determine if property defined"""
        if isinstance(self.state, tuple):
            return getattr(self.state, attr, None) is not None
        return attr in self.state

    @property
    def pressure(self):
        return self._get("pressure")

    @property
    def pressures(self):
        return self._get("pressures")

    def _get(self, attr):
        if isinstance(self.state, tuple):
            return getattr(self.state, attr, None)
        return self.state.get(attr, None)
