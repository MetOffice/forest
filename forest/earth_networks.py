import os
import glob
import datetime as dt
import pandas as pd
from forest import geo
import bokeh.models


class View(object):
    def __init__(self, loader):
        self.loader = loader
        self.source = bokeh.models.ColumnDataSource({
            "x": [],
            "y": [],
            "date": [],
            "longitude": [],
            "latitude": [],
            "flash_type": []
        })

    def render(self, valid_date):
        frame = self.loader.load_date(valid_date)
        x, y = geo.web_mercator(
                frame.longitude,
                frame.latitude)
        self.source.data = {
            "x": x,
            "y": y,
            "date": frame.date,
            "longitude": frame.longitude,
            "latitude": frame.latitude,
            "flash_type": frame.flash_type,
        }

    def add_figure(self, figure):
        renderer = figure.circle(
                x="x",
                y="y",
                size=10,
                source=self.source)
        tool = bokeh.models.HoverTool(
                tooltips=[
                    ('Time', '@date{%F}'),
                    ('Lon', '@longitude'),
                    ('Lat', '@latitude'),
                    ('Flash type', '@flash_type')],
                formatters={
                    'date': 'datetime'
                },
                renderers=[renderer])
        figure.add_tools(tool)
        return renderer


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
        end = start + dt.timedelta(minutes=15)
        s = "{:%Y-%m-%dT%H:%M}".format(start)
        e = "{:%Y-%m-%dT%H:%M}".format(end)
        small_frame = frame[s:e].copy()
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
