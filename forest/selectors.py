"""Indirect access to State properties"""
import datetime as dt


class Selector:
    """Access data stored in state"""
    _props = [
        "pressure",
        "pressures",
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

    @property
    def valid_time(self):
        if isinstance(self.state, tuple):
            return self.asdatetime(self.state.valid_time)
        return self.asdatetime(self.state.get("valid_time", None))

    @property
    def initial_time(self):
        if isinstance(self.state, tuple):
            return self.asdatetime(self.state.initial_time)
        return self.asdatetime(self.state.get("initial_time", None))

    @staticmethod
    def asdatetime(value):
        if value is None:
            return value
        if isinstance(value, str):
            pattern = "%Y-%m-%d %H:%M:%S"
            return dt.datetime.strptime(value, pattern)
        return value

    def __getattr__(self, attr):
        if attr in self._props:
            return self._get(attr)
        return self.__dict__.get(attr)

    def _get(self, attr):
        if isinstance(self.state, tuple):
            return getattr(self.state, attr, None)
        return self.state.get(attr, None)
