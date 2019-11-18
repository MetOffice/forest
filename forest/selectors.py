"""Indirect access to State properties"""


class Selector:
    def __init__(self, state):
        self.state = state

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
