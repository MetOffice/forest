import re
import os
import glob
import datetime as dt
import datashader
import pandas as pd
from functools import lru_cache
from forest import geo
from forest.util import to_datetime as _to_datetime
import forest.util
from forest.old_state import old_state, unique
import bokeh.models
import bokeh.palettes
import bokeh.colors
import numpy as np


class Dataset:
    """High-level class to relate navigators, loaders and views"""
    def __init__(self, pattern=None, **kwargs):
        self.pattern = pattern
        self.loader = Loader()
        self.locator = TimestampLocator(pattern)

    def navigator(self):
        """Construct navigator"""
        return Navigator(self.locator)

    def map_view(self):
        """Construct view"""
        return View(self.loader, self.locator)


class View:
    def __init__(self, loader, locator):
        self.loader = loader
        self.locator = locator
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
        self.hover_tools = {
            "image": []
        }
        self.color_mappers = {}
        self.color_mappers["image"] = bokeh.models.LinearColorMapper(
            low=0,
            high=1,
            palette="Inferno256",
            nan_color=bokeh.colors.RGB(0, 0, 0, a=0)
        )
        self.sources = {}
        self.sources["scatter"] = bokeh.models.ColumnDataSource(self.empty_image)
        self.sources["image"] = bokeh.models.ColumnDataSource({
            "x": [],
            "y": [],
            "dw": [],
            "dh": [],
            "image": [],
        })
        self.variable_to_method = {
            "Lightning": self.scatter,
        }

    @old_state
    @unique
    def render(self, state):
        if state.valid_time is None:
            return
        if state.variable is None:
            return
        self.variable_to_method.get(state.variable, self.image)(state)

    def image(self, state):
        """Image colored by time since flash or flash density"""
        valid_time =_to_datetime(state.valid_time)

        # 15 minute/1 hour slice of data?
        window = dt.timedelta(minutes=60)  # 1 hour window
        paths = self.locator.find_period(valid_time, window)
        frame = self.loader.load(paths)
        frame = self.select_date(frame, valid_time, window)

        # Filter intra-cloud/cloud-ground rows
        if "intra-cloud" in state.variable.lower():
            frame = frame[frame["flash_type"] == "IC"]
        elif "cloud-ground" in state.variable.lower():
            frame = frame[frame["flash_type"] == "CG"]

        # EarthNetworks validity box (not needed if tiling algorithm)
        longitude_range = (26, 40)
        latitude_range = (-12, 4)
        x_range, y_range = geo.web_mercator(longitude_range,
                                            latitude_range)

        x, y = geo.web_mercator(frame["longitude"], frame["latitude"])
        frame["x"] = x
        frame["y"] = y
        pixels = 256
        canvas = datashader.Canvas(
            plot_width=pixels,
            plot_height=pixels,
            x_range=x_range,
            y_range=y_range
        )

        if "density" in state.variable.lower():
            # N flashes per pixel
            agg = canvas.points(frame, "x", "y", datashader.count())
        else:
            frame["since_flash"] = self.since_flash(frame["date"], valid_time)
            agg = canvas.points(frame, "x", "y", datashader.max("since_flash"))

        # Note: DataArray objects are not JSON serializable, .values is the
        #       same data cast as a numpy array
        x = agg.x.values.min()
        y = agg.y.values.min()
        dw = agg.x.values.max() - x
        dh = agg.y.values.max() - y
        image = np.ma.masked_array(agg.values.astype(np.float),
                                   mask=np.isnan(agg.values))
        if "density" in state.variable.lower():
            image[image == 0] = np.ma.masked # Remove pixels with no data

        # Update color_mapper
        color_mapper = self.color_mappers["image"]
        if "density" in state.variable.lower():
            color_mapper.palette = bokeh.palettes.all_palettes["Spectral"][8]
            color_mapper.low = 0
            color_mapper.high = agg.values.max()
        else:
            color_mapper.palette = bokeh.palettes.all_palettes["RdGy"][8]
            color_mapper.low = 0
            color_mapper.high = 60 * 60 # 1 hour

        # Update tooltips
        for hover_tool in self.hover_tools["image"]:
            hover_tool.tooltips = self.tooltips(state.variable)
            hover_tool.formatters = self.formatters(state.variable)

        if "density" in state.variable.lower():
            units = "events"
        else:
            units = "seconds"

        data = {
            "x": [x],
            "y": [y],
            "dw": [dw],
            "dh": [dh],
            "image": [image],
        }
        meta_data = {
            "variable": [state.variable],
            "date": [valid_time],
            "units": [units],
            "window": [window.total_seconds()]
        }
        data.update(meta_data)
        self.sources["image"].data = data

    def scatter(self, state):
        """Scatter plot of flash position colored by time since flash"""
        valid_time = _to_datetime(state.valid_time)
        paths = self.locator.find(valid_time)
        frame = self.loader.load(paths)
        frame = self.select_date(frame, valid_time)
        frame = frame[:400]  # Limit points
        frame['time_since_flash'] = self.since_flash(frame['date'], valid_time)
        if len(frame) == 0:
            return self.empty_image
        x, y = geo.web_mercator(
                frame.longitude,
                frame.latitude)
        self.color_mapper.low = np.min(frame.time_since_flash)
        self.color_mapper.high = np.max(frame.time_since_flash)
        self.sources["scatter"].data = {
            "x": x,
            "y": y,
            "date": frame.date,
            "longitude": frame.longitude,
            "latitude": frame.latitude,
            "flash_type": frame.flash_type,
            "time_since_flash": frame.time_since_flash
        }

    def select_date(self, frame, date, window):
        if len(frame) == 0:
            return frame
        frame = frame.set_index('date')
        start = date
        end = start + window
        s = "{:%Y-%m-%dT%H:%M}".format(start)
        e = "{:%Y-%m-%dT%H:%M}".format(end)
        small_frame = frame[s:e].copy()
        return small_frame.reset_index()

    def since_flash(self, date_column, date):
        """Pandas helper to calculate seconds since valid date"""
        if len(date_column) == 0:
            return []
        if isinstance(date, str):
            date = pd.Timestamp(date)
        if isinstance(date_column, list):
            date_column = pd.Series(pd.to_datetime(date_column))
        return (date_column - date).dt.total_seconds()

    @staticmethod
    def tooltips(variable):
        if "density" in variable.lower():
            return [
                ('Variable', '@variable'),
                ('Time window', '@window{00:00:00}'),
                ('Period start', '@date{%Y-%m-%d %H:%M:%S}'),
                ('Value', '@image @units')]
        else:
            return [
                ('Variable', '@variable'),
                ('Time window', '@window{00:00:00}'),
                ('Period start', '@date{%Y-%m-%d %H:%M:%S}'),
                ('Since start', '@image{00:00:00}')]

    @staticmethod
    def formatters(variable):
        defaults = {
            "@date": "datetime",
            "@window": "numeral"
        }
        if "density" in variable.lower():
            return defaults
        else:
            return {**defaults, **{'@image': 'numeral'}}

    def add_figure(self, figure):
        renderer = figure.cross(
                x="x",
                y="y",
                size=10,
                fill_color={'field': 'time_since_flash', 'transform': self.color_mapper},
                line_color={'field': 'time_since_flash', 'transform': self.color_mapper},
                source=self.sources["scatter"])

        # Add image glyph_renderer
        renderer = figure.image(x="x", y="y", dw="dw", dh="dh", image="image",
                     source=self.sources["image"],
                     color_mapper=self.color_mappers["image"])
        custom_js = bokeh.models.CustomJS(
            args=dict(source=self.sources["image"]), code="""
            let variable = source.data['variable'][0]
            let idx = cb_data.index.image_indices[0]
            if (typeof idx !== 'undefined') {
                let number = source.data['image'][0][idx.flat_index]
                if (typeof number === "undefined") {
                    number = source.data['image'][0][idx.dim1][idx.dim2]
                }
                if (isNaN(number)) {
                    if (typeof window._tooltips === 'undefined') {
                        window._tooltips = {}
                    }
                    if (typeof window._tooltips[variable] === 'undefined') {
                        // TODO: Remove global variable
                        window._tooltips[variable] = cb_obj.tooltips
                    }
                    cb_obj.tooltips = [
                        ['Variable', '@variable'],
                    ];
                } else {
                    cb_obj.tooltips = window._tooltips[variable]
                }
            }
        """)
        variable = "Strike density (cloud-ground)"
        tool = bokeh.models.HoverTool(
                tooltips=self.tooltips(variable),
                formatters=self.formatters(variable),
                renderers=[renderer],
                callback=custom_js
        )
        self.hover_tools["image"].append(tool)
        figure.add_tools(tool)
        return renderer


class TimestampLocator:
    """Find files by time stamp"""
    def __init__(self, pattern):
        if pattern is None:
            paths = []
        else:
            paths = glob.glob(pattern)

        # TODO: Find better way to reduce data volume
        if len(paths) > 1000:
            paths = sorted(paths)[-1000:]
        self.paths = paths

        self.table = {}
        for path in self.paths:
            self.table[self._parse_date(path)] = path

        times = [
            self._parse_date(path) for path in self.paths
        ]
        times = [t for t in times if t is not None]
        index = pd.DatetimeIndex(times)
        self._valid_times = index.sort_values()

    def find_period(self, date, window):
        return self.find(date)  # TODO: implement search window

    def find(self, date):
        if date in self.table:
            return [self.table[date]]
        else:
            return []

    @staticmethod
    def _parse_date(path):
        groups = re.search(r"[0-9]{8}T[0-9]{4}", os.path.basename(path))
        if groups is not None:
            return dt.datetime.strptime(groups[0], "%Y%m%dT%H%M")

    def valid_times(self):
        if len(self._valid_times) == 0:
            return []
        return self._valid_times


class Navigator:
    """Meta-data needed to navigate the dataset"""
    def __init__(self, locator):
        self.locator = locator

    def variables(self, pattern):
        labels = []
        for metric in ("time since flash", "strike density"):
            for category in ("cloud-ground", "intra-cloud", "total"):
                label = "{} ({})".format(metric.capitalize(), category)
                labels.append(label)
        return labels

    def initial_times(self, pattern, variable):
        return [dt.datetime(1970, 1, 1)]

    def valid_times(self, pattern, variable, initial_time):
        # Populates initial_state and used by forest.db.control.Control
        return self.locator.valid_times()

    def pressures(self, pattern, variable, initial_time):
        return []


class Loader:
    """Methods to manipulate EarthNetworks data"""
    def load(self, csv_files):
        if isinstance(csv_files, str):
            csv_files = [csv_files]
        frames = []
        for csv_file in csv_files:
            frame = self.load_file(csv_file)
            frames.append(frame)
        if len(frames) == 0:
            return pd.DataFrame({
                "flash_type": [],
                "date": [],
                "latitude": [],
                "longitude": [],
            })
        else:
            return pd.concat(frames, ignore_index=True)

    @lru_cache(maxsize=32)
    def load_file(self, path):
        return pd.read_csv(
            path,
            parse_dates=[1],
            converters={0: self.flash_type},
            usecols=[0, 1, 2, 3],
            names=["flash_type", "date", "latitude", "longitude"],
            header=None)

    @staticmethod
    def flash_type(value):
        return {
            "0": "CG",
            "1": "IC",
            "9": "Keep alive"
        }.get(value, value)
