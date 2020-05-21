import re
import sqlite3
import os
import glob
import datetime as dt
import bokeh.models
import xarray
import numpy as np
from functools import lru_cache
from forest.exceptions import FileNotFound, IndexNotFound
from forest.old_state import old_state, unique
import forest.util
import forest.map_view
from forest import (
        geo,
        locate)


ENGINE = "h5netcdf"
MIN_DATETIME64 = np.datetime64('0001-01-01T00:00:00.000000')


def _natargmax(arr):
    """ Find the arg max when an array contains NaT's"""
    no_nats = np.where(np.isnat(arr), MIN_DATETIME64, arr)
    return np.argmax(no_nats)


class Dataset:
    def __init__(self, pattern=None, database_path=None, **kwargs):
        self.pattern = pattern
        if database_path is None:
            database_path = ":memory:"
        self.database = Database(database_path)
        self.locator = Locator(self.pattern, self.database)

    def navigator(self):
        return Navigator(self.locator, self.database)

    def map_view(self, color_mapper):
        loader = Loader(self.locator)
        return forest.map_view.map_view(loader, color_mapper, use_hover_tool=False)


class Database:
    """Meta-data store for EIDA50 dataset"""
    def __init__(self, path=":memory:"):
        self.fmt = "%Y-%m-%d %H:%M:%S"
        self.path = path
        self.connection = sqlite3.connect(self.path)
        self.cursor = self.connection.cursor()

        # Schema
        query = """
            CREATE TABLE IF NOT EXISTS file (
                      id INTEGER PRIMARY KEY,
                    path TEXT,
                         UNIQUE(path));
        """
        self.cursor.execute(query)
        query = """
            CREATE TABLE IF NOT EXISTS time (
                      id INTEGER PRIMARY KEY,
                    time TEXT,
                 file_id INTEGER,
                 FOREIGN KEY(file_id) REFERENCES file(id));
        """
        self.cursor.execute(query)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close connection gracefully"""
        self.connection.commit()
        self.connection.close()

    def insert_times(self, times, path):
        """Store times"""
        # Update file table
        query = """
            INSERT OR IGNORE INTO file (path) VALUES (:path);
        """
        self.cursor.execute(query, {"path": path})
        self.connection.commit()

        # Update time table
        query = """
            INSERT INTO time (time, file_id)
                 VALUES (:time, (SELECT id FROM file WHERE path = :path));
        """
        texts = [time.strftime(self.fmt) for time in times]
        args = [{"path": path, "time": text} for text in texts]
        self.cursor.executemany(query, args)
        self.connection.commit()

    def fetch_times(self, path=None):
        """Retrieve times"""
        if path is None:
            query = """
                SELECT time
                  FROM time;
            """
            rows = self.cursor.execute(query)
        else:
            query = """
                SELECT time.time
                  FROM time
                  JOIN file
                    ON file.id = time.file_id
                 WHERE file.path = :path;
            """
            rows = self.cursor.execute(query, {"path": path})
        rows = self.cursor.fetchall()
        texts = [text for text, in rows]
        times = [dt.datetime.strptime(text, self.fmt) for text in texts]
        return list(sorted(times))

    def fetch_paths(self):
        """Retrieve paths"""
        query = """
            SELECT path FROM file;
        """
        rows = self.cursor.execute(query).fetchall()
        texts = [text for text, in rows]
        return list(sorted(texts))


class Locator:
    """Locate EIDA50 satellite images"""
    def __init__(self, pattern, database):
        self.pattern = pattern
        self._glob = forest.util.cached_glob(dt.timedelta(minutes=15))
        self.database = database

    def all_times(self, paths):
        """All available times"""
        # Parse times from file names not in database
        unparsed_paths = []
        filename_times = []
        database_paths = set(self.database.fetch_paths())
        for path in paths:
            if path in database_paths:
                continue
            time = self.parse_date(path)
            if time is None:
                unparsed_paths.append(path)
                continue
            filename_times.append(time)

        # Store unparsed files in database
        for path in unparsed_paths:
            times = self.load_time_axis(path)  # datetime64[s]
            self.database.insert_times(times.astype(dt.datetime), path)

        # All recorded times
        database_times = self.database.fetch_times()

        # Combine database/timestamp information
        arrays = [
            np.array(database_times, dtype='datetime64[s]'),
            np.array(filename_times, dtype='datetime64[s]')]
        return np.unique(np.concatenate(arrays))

    def find(self, paths, date):
        """Find file and index related to date

        .. note:: Find should not write to disk
        """
        # Search file system
        path = self.find_file(paths, date)

        # Load times from database
        if path in self.database.fetch_paths():
            times = self.database.fetch_times(path)
            times = np.array(times, dtype='datetime64[s]')
        else:
            times = self.load_time_axis(path)  # datetime64[s]

        index = self.find_index(times, date, dt.timedelta(minutes=15))
        return path, index

    def glob(self):
        return self._glob(self.pattern)

    @staticmethod
    @lru_cache()
    def load_time_axis(path):
        with xarray.open_dataset(path, engine=ENGINE) as nc:
            values = nc["time"]
        return np.array(values, dtype='datetime64[s]')

    def find_file(self, paths, user_date):
        """"Find file likely to contain user supplied date

        .. note:: Search based on timestamp only

        .. warning:: Not suitable for searching unparsable file names
        """
        if isinstance(user_date, (dt.datetime, str)):
            user_date = np.datetime64(user_date, 's')
        dates = np.array([
            self.parse_date(path) for path in paths],
            dtype='datetime64[s]')
        mask = ~(dates <= user_date)
        if mask.all():
            msg = "No file for {}".format(user_date)
            raise FileNotFound(msg)
        before_dates = np.ma.array(
                dates, mask=mask, dtype='datetime64[s]')
        i =  _natargmax(before_dates.filled())
        return paths[i]

    @staticmethod
    def find_index(times, time, length):
        """Search for index inside array of datetime64[s] values"""
        dtype = 'datetime64[s]'
        if isinstance(times, list):
            times = np.asarray(times, dtype=dtype)
        if isinstance(time, (dt.datetime, str)):
            time = np.datetime64(time, 's')
        bounds = locate.bounds(times, length)
        inside = locate.in_bounds(bounds, time)
        valid_times = np.ma.array(times, mask=~inside)
        if valid_times.mask.all():
            msg = "{}: not found".format(time)
            raise IndexNotFound(msg)
        return _natargmax(valid_times.filled())

    @staticmethod
    def parse_date(path):
        """Parse timestamp into datetime or None"""
        for regex, fmt in [
                (r"([0-9]{8})\.nc", "%Y%m%d"),
                (r"([0-9]{8}T[0-9]{4}Z)\.nc", "%Y%m%dT%H%MZ")]:
            groups = re.search(regex, path)
            if groups is None:
                continue
            else:
                return dt.datetime.strptime(groups[1], fmt)


class Loader:
    def __init__(self, locator):
        self.locator = locator
        self.empty_image = {
            "x": [],
            "y": [],
            "dw": [],
            "dh": [],
            "image": []
        }
        self.cache = {}
        paths = self.locator.glob()
        if len(paths) > 0:
            with xarray.open_dataset(paths[-1], engine=ENGINE) as nc:
                self.cache["longitude"] = nc["longitude"].values
                self.cache["latitude"] = nc["latitude"].values

    @property
    def longitudes(self):
        return self.cache["longitude"]

    @property
    def latitudes(self):
        return self.cache["latitude"]

    def image(self, state):
        if state.valid_time is None:
            data = self.empty_image
        else:
            try:
                data = self._image(forest.util.to_datetime(state.valid_time))
            except (FileNotFound, IndexNotFound):
                data = self.empty_image
        return data

    def _image(self, valid_time):
        paths = self.locator.glob()
        path, itime = self.locator.find(paths, valid_time)
        return self.load_image(path, itime)

    def load_image(self, path, itime):
        lons = self.longitudes
        lats = self.latitudes
        with xarray.open_dataset(path, engine=ENGINE) as nc:
            values = nc["data"][itime].values

        # Use datashader to coarsify images from 4.4km to 8.8km grid
        scale = 2
        return geo.stretch_image(
                lons, lats, values,
                plot_width=int(values.shape[1] / scale),
                plot_height=int(values.shape[0] / scale))


class Navigator:
    """Facade to map Navigator API to Locator"""
    def __init__(self, locator, database):
        self.locator = locator
        self.database = database

    def __call__(self, store, action):
        """Middleware interface"""
        import forest.db.control
        kind = action["kind"]
        if kind == forest.db.control.SET_VALUE:
            key, time = (action["payload"]["key"],
                         action["payload"]["value"])
            if key == "valid_time":
                # Detect missing file
                paths = self.locator.glob()
                path = self.locator.find_file(paths, time)
                if path not in self.database.fetch_paths():
                    times = self.locator.load_time_axis(path)  # datetime64[s]
                    self.database.insert_times(times.astype(dt.datetime), path)

                # Update stale state
                store_times = store.state.get("valid_times", [])
                database_times = self._times()
                if len(store_times) != len(database_times):
                    yield forest.db.control.set_value(
                        "valid_times", database_times)
        yield action

    def variables(self, pattern):
        return ["EIDA50"]

    def initial_times(self, pattern, variable):
        return [dt.datetime(1970, 1, 1)]

    def valid_times(self, pattern, variable, initial_time):
        """Get available times given application state"""
        return self._times()

    def _times(self):
        """Get available times given user selected valid_time

        :param valid_time: application state valid time
        """
        paths = self.locator.glob()
        datetimes = self.locator.all_times(paths)
        return np.array(datetimes, dtype='datetime64[s]')

    def pressures(self, pattern, variable, initial_time):
        return []
