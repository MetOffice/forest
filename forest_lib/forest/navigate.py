"""Navigation tools for Forest

Streams of +1 and -1 can be accumulated and processed to
index arrays of times, hours and model run times

Streams can be merged and combined to update the application state

I/O can be triggered when appropriate to detect available times

"""
import bokeh.layouts
import numpy as np
import datetime as dt
from collections import namedtuple


__all__ = [
    "Forecast"
]



class Forecast(object):
    """Point in forecast/real time space"""
    __slots__ = ("start", "length")
    def __init__(self, start, length):
        self.start = start
        self.length = length

    @classmethod
    def stamp(cls, text, hours, fmt="%Y-%m-%d %H:%M"):
        """create from formatted string and int"""
        return cls(dt.datetime.strptime(text, fmt),
                   dt.timedelta(hours=hours))

    @property
    def valid_time(self):
        return self.start + self.length

    def label(self, fmt="{:%Y-%m-%d %H:%M} T{:+}"):
        """string format of object"""
        return fmt.format(self.start, int(self.hours))

    @property
    def hours(self):
        return self.length.total_seconds() / (60. * 60.)

    def __repr__(self):
        pattern = "{}(start='{}', length='{}')"
        return pattern.format(self.__class__.__name__,
                              self.start,
                              self.length)

    def __eq__(self, other):
        return ((self.start == other.start) &
                (self.length == other.length))


def forecast_view(stream):
    """Controls to navigate forecasts

    :returns: WidgetBox
    """
    p = bokeh.models.Paragraph()
    text_stream = stream.map(str).map(text(p))
    return bokeh.layouts.WidgetBox(p,
                                   plus_button(stream),
                                   minus_button(stream))


def plus_button(stream):
    btn = bokeh.models.Button(label="+")
    btn.on_click(emit(stream, +1))
    return btn


def minus_button(stream):
    btn = bokeh.models.Button(label="-")
    btn.on_click(emit(stream, -1))
    return btn


def emit(stream, value):
    """Creates a closure for on_click"""
    def closure():
        stream.emit(value)
    return closure


def text(widget):
    """A widget .text assignment closure suitable for notify"""
    def closure(value):
        widget.text = value
    return closure


def forecast_label(initial_time, forecast_length):
    """Forecast naming convention"""
    if isinstance(forecast_length, dt.timedelta):
        forecast_length = int(hours(forecast_length))
    return "{:%Y-%m-%d %H:%M} T{:+}".format(initial_time,
                                            forecast_length)


def hours(delta):
    """Estimate hours from timedelta object"""
    return delta.total_seconds() / (60. * 60.)


def forecasts(times):
    """Convert datetimes to Forecast tuples"""
    start = initial_time(times)
    hours = forecast_lengths(times)
    return [Forecast(start, hour) for hour in hours]


def forecast_lengths(times):
    """Estimate forecast lengths from valid times"""
    return np.asarray(times, dtype=object) - times[0]


def initial_time(times):
    """Estimate forecast initialisation time from valid times"""
    return times[0]
