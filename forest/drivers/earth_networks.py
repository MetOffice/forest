import re
import os
import glob
import datetime as dt
import pandas as pd
from forest import geo
from forest.util import to_datetime as _to_datetime
from forest.old_state import old_state, unique
import bokeh.models
import bokeh.palettes
import numpy as np


class Dataset:
    """High-level class to relate navigators, loaders and views"""
    def __init__(self, pattern=None, **kwargs):
        self.pattern = pattern
        if pattern is not None:
            self._paths = glob.glob(pattern)
        else:
            self._paths = []

    def navigator(self):
        """Construct navigator"""
        return Navigator(self._paths)

    def map_view(self):
        """Construct view"""
        return View(Loader(self._paths))


class View(object):
    def __init__(self, loader):
        self.loader = loader
        palette = bokeh.palettes.all_palettes['Spectral'][11][::-1]
        self.color_mapper = bokeh.models.LinearColorMapper(low=-1000, high=0, palette=palette)
        self.empty_image = {
            "x": [],
            "y": [],
            "date": [],
            "longitude": [],
            "latitude": [],
            "flash_type": [],
            "time_since_flash": []
        }
        self.source = bokeh.models.ColumnDataSource(self.empty_image)

    @old_state
    @unique
    def render(self, state):
        if state.valid_time is None:
            return

        valid_time = _to_datetime(state.valid_time)
        frame = self.loader.load_date(valid_time)
        if len(frame) == 0:
            return self.empty_image
        x, y = geo.web_mercator(
                frame.longitude,
                frame.latitude)
        self.color_mapper.low = np.min(frame.time_since_flash)
        self.color_mapper.high = np.max(frame.time_since_flash)
        self.source.data = {
            "x": x,
            "y": y,
            "date": frame.date,
            "longitude": frame.longitude,
            "latitude": frame.latitude,
            "flash_type": frame.flash_type,
            "time_since_flash": frame.time_since_flash
        }

    def add_figure(self, figure):
        renderer = figure.cross(
                x="x",
                y="y",
                size=10,
                fill_color={'field': 'time_since_flash', 'transform': self.color_mapper},
                line_color={'field': 'time_since_flash', 'transform': self.color_mapper},
                source=self.source)
        tool = bokeh.models.HoverTool(
                tooltips=[
                    ('Time', '@date{%F}'),
                    ('Since flash', '@time_since_flash'),
                    ('Lon', '@longitude'),
                    ('Lat', '@latitude'),
                    ('Flash type', '@flash_type')],
                formatters={
                    'date': 'datetime'
                },
                renderers=[renderer])
        figure.add_tools(tool)
        return renderer


class Navigator:
    def __init__(self, paths):
        self.paths = paths
        times = [
            self._parse_date(path) for path in paths
        ]
        times = [t for t in times if t is not None]
        self._valid_times = list(sorted(set(times)))

    @staticmethod
    def _parse_date(path):
        groups = re.search(r"[0-9]{8}T[0-9]{4}", os.path.basename(path))
        if groups is not None:
            return dt.datetime.strptime(groups[0], "%Y%m%dT%H%M")

    def variables(self, pattern):
        return ["Lightning"]

    def initial_times(self, pattern, variable):
        return [dt.datetime(1970, 1, 1)]

    def valid_times(self, pattern, variable, initial_time):
        return self._valid_times

    def pressures(self, pattern, variable, initial_time):
        return []


class Loader(object):
    def __init__(self, paths):
        self.paths = paths
        if len(self.paths) > 0:
            self.frame = self.read(paths)

    @classmethod
    def pattern(cls, text):
        return cls(list(sorted(glob.glob(os.path.expanduser(text)))))

    def load_date(self, date):
        frame = self.frame.set_index('date')
        start = date
        end = start + dt.timedelta(minutes=60)  # 1 hour window
        s = "{:%Y-%m-%dT%H:%M}".format(start)
        e = "{:%Y-%m-%dT%H:%M}".format(end)
        small_frame = frame[s:e].copy()
        small_frame['time_since_flash'] = [t.total_seconds() for t in date - small_frame.index]
        return small_frame.reset_index()

    @staticmethod
    def read(csv_files):
        if isinstance(csv_files, str):
            csv_files = [csv_files]
        frames = []
        for csv_file in csv_files:
            frame = pd.read_csv(
                csv_file,
                parse_dates=[1],
                converters={0: Loader.flash_type},
                usecols=[0, 1, 2, 3],
                names=["flash_type", "date", "latitude", "longitude"],
                header=None)
            frames.append(frame)
        if len(frames) == 0:
            return None
        else:
            return pd.concat(frames, ignore_index=True)

    @staticmethod
    def flash_type(value):
        return {
            "0": "CG",
            "1": "IC",
            "9": "Keep alive"
        }.get(value, value)
