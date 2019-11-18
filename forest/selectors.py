"""Indirect access to State properties"""


class Selector:
    def __init__(self, state):
        self.state = state

    @property
    def pressure(self):
        if isinstance(self.state, tuple):
            return self.state.pressure
        return self.state.get("pressure", None)
