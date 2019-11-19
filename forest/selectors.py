"""Indirect access to State properties"""
import datetime as dt
import numpy as np
import cftime


class Selector:
    """Access data stored in state

    Provides interface for other consumers of state to
    access values stored in state
    """
    def __init__(self, state):
        self.state = state

    def defined(self, attr):
        """Determine if property defined"""
        return attr in self.state

    @property
    def pressure(self):
        return self.state.get("pressure", None)

    @property
    def pressures(self):
        return self.state.get("pressures", None)

    @property
    def variable(self):
        return self.state.get("variable", None)

    @property
    def valid_time(self):
        value = self.state.get("valid_time", None)
        if value is None:
            return
        return self.to_datetime(value)

    @property
    def initial_time(self):
        value = self.state.get("initial_time", None)
        if value is None:
            return
        return self.to_datetime(value)

    @staticmethod
    def to_datetime(d):
        if isinstance(d, dt.datetime):
            return d
        if isinstance(d, cftime.DatetimeNoLeap):
            return datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
        elif isinstance(d, str):
            try:
                return dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return dt.datetime.strptime(d, "%Y-%m-%dT%H:%M:%S")
        elif isinstance(d, np.datetime64):
            return d.astype(dt.datetime)
        else:
            raise Exception("Unknown value: {}".format(d))
