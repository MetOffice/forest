import os
import re
import datetime as dt
from functools import partial
import scipy.ndimage
import numpy as np


class Observable(object):
    def __init__(self):
        self.uid = 0
        self.listeners = []

    def subscribe(self, listener):
        self.uid += 1
        self.listeners.append(listener)
        return partial(self.unsubscribe, int(self.uid))

    def unsubscribe(self, uid):
        del self.listeners[uid]

    def announce(self, *args):
        for listener in self.listeners:
            listener(*args)


def select(dropdown):
    def wrapped(new):
        for label, value in dropdown.menu:
            if value == new:
                dropdown.label = label
    return wrapped


def timeout_cache(interval):
    def decorator(f):
        cache = {}
        call_time = {}
        def wrapped(x):
            nonlocal cache
            nonlocal call_time
            now = dt.datetime.now()
            if x not in cache:
                cache[x] = f(x)
                call_time[x] = now
                return cache[x]
            else:
                if (now - call_time[x]) > interval:
                    cache[x] = f(x)
                    call_time[x] = now
                    return cache[x]
                else:
                    return cache[x]
        return wrapped
    return decorator


def coarsify(lons, lats, values, fraction):
    values = scipy.ndimage.zoom(values, fraction)
    ny, nx = values.shape
    lons = np.linspace(lons.min(), lons.max(), nx)
    lats = np.linspace(lats.min(), lats.max(), ny)
    return lons, lats, values


def initial_time(path):
    name = os.path.basename(path)
    groups = re.search(r"[0-9]{8}T[0-9]{4}Z", path)
    if groups:
        return dt.datetime.strptime(groups[0], "%Y%m%dT%H%MZ")
