import os
from functools import lru_cache
import numpy as np
from .connection import Connection
from forest.exceptions import SearchFail
from forest import mark


__all__ = [
    "Locator"
]


class Locator(Connection):
    """Query database for path and index related to fields"""
    def __init__(self, connection, directory=None):
        self.directory = directory
        self.connection = connection
        self.cursor = self.connection.cursor()

    @mark.sql_sanitize_time("initial_time", "valid_time")
    def locate(
            self,
            pattern,
            variable,
            initial_time,
            valid_time,
            pressure=None,
            tolerance=0.001):
        valid_time64 = np.datetime64(valid_time, 's')
        for file_name in self.file_names(
                pattern,
                variable,
                initial_time,
                valid_time):
            if self.directory is not None:
                # HACK: consider refactor
                path = os.path.join(self.directory, os.path.basename(file_name))
            else:
                path = file_name
            ta, pa = self.axes(file_name, variable)
            if (ta is None) and (pa is None):
                return path, ()
            elif (ta is None) and (pa is not None):
                if pressure is None:
                    raise SearchFail("Need pressure to search pressure axis")
                pressures = self.coordinate(file_name, variable, "pressure")
                i = np.where(np.abs(pressures - pressure) < tolerance)[0][0]
                return path, (i,)
            elif (ta is not None) and (pa is None):
                times = self.coordinate(file_name, variable, "time")
                i = np.where(times == valid_time64)[0][0]
                return path, (i,)
            elif (ta is not None) and (pa is not None):
                if pressure is None:
                    raise SearchFail("Need pressure to search pressure axis")
                times = self.coordinate(file_name, variable, "time")
                pressures = self.coordinate(file_name, variable, "pressure")
                if (ta == 0) and (pa == 0):
                    pts = np.where(
                        (times == valid_time64) &
                        (np.abs(pressures - pressure) < tolerance))
                    i = pts[0][0]
                    return path, (i,)
                else:
                    ti = np.where(times == valid_time64)[0][0]
                    pi = np.where(np.abs(pressures - pressure) < tolerance)[0][0]
                    return path, (ti, pi)
        raise SearchFail("Could not locate: {}".format(pattern))

    @mark.sql_sanitize_time("initial_time", "valid_time")
    @lru_cache()
    def file_names(self, pattern, variable, initial_time, valid_time):
        self.cursor.execute("""
            SELECT DISTINCT(f.name)
              FROM file AS f
              JOIN variable AS v
                ON v.file_id = f.id
              JOIN variable_to_time AS vt
                ON vt.variable_id = v.id
              JOIN time AS t
                ON t.id = vt.time_id
             WHERE f.name GLOB :pattern
               AND f.reference = :initial_time
               AND v.name = :variable
               AND t.value = :valid_time
        """, dict(
            pattern=pattern,
            variable=variable,
            initial_time=initial_time,
            valid_time=valid_time,
        ))
        return [file_name for file_name, in self.cursor.fetchall()]

    @lru_cache()
    def coordinate(self, file_name, variable, coord):
        if coord == "pressure":
            self.cursor.execute("""
                SELECT p.i, p.value
                  FROM file AS f
                  JOIN variable AS v
                    ON v.file_id = f.id
                  JOIN variable_to_pressure AS vp
                    ON vp.variable_id = v.id
                  JOIN pressure AS p
                    ON p.id = vp.pressure_id
                 WHERE f.name = :file_name
                   AND v.name = :variable
              ORDER BY p.i
            """, dict(
                file_name=file_name,
                variable=variable
            ))
            rows = self.cursor.fetchall()
        elif coord == "time":
            self.cursor.execute("""
                SELECT t.i, t.value
                  FROM file AS f
                  JOIN variable AS v
                    ON v.file_id = f.id
                  JOIN variable_to_time AS vt
                    ON vt.variable_id = v.id
                  JOIN time AS t
                    ON t.id = vt.time_id
                 WHERE f.name = :file_name
                   AND v.name = :variable
              ORDER BY t.i
            """, dict(
                file_name=file_name,
                variable=variable
            ))
            rows = self.cursor.fetchall()
        else:
            raise Exception("unknown coordinate: {}".format(coord))
        if coord == "time":
            dtype = "datetime64[s]"
        else:
            dtype = "f"
        index, values = zip(*rows)
        array = np.empty(np.max(index) + 1, dtype=dtype)
        for i, v in zip(index, values):
            array[i] = v
        return array

    @lru_cache()
    def axes(self, file_name, variable):
        """Time/pressure axis information

        :returns: (time_axis, pressure_axis)
        """
        self.cursor.execute("""
            SELECT v.time_axis, v.pressure_axis
              FROM file AS f
              JOIN variable AS v
                ON v.file_id = f.id
             WHERE f.name = :file_name
               AND v.name = :variable
        """, dict(
            file_name=file_name,
            variable=variable
        ))
        return self.cursor.fetchone()
