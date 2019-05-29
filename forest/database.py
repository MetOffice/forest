import sqlite3
import iris


class Database(object):
    def __init__(self, connection):
        self.connection = connection
        self.cursor = self.connection.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS file (
                      id INTEGER PRIMARY KEY,
                    name TEXT,
            initial_time TEXT,
                         UNIQUE(name))
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS variable (
                      id INTEGER PRIMARY KEY,
                    name TEXT,
                 file_id INTEGER,
                         FOREIGN KEY(file_id) REFERENCES file(id))
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS coordinate (
                      id INTEGER PRIMARY KEY,
                    name TEXT,
                    axis INTEGER,
             variable_id INTEGER,
                         FOREIGN KEY(variable_id) REFERENCES variable(id))
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS pressure (
                      id INTEGER PRIMARY KEY,
                       i INTEGER,
                   value REAL)
        """)

    @classmethod
    def connect(cls, file_name):
        return cls(sqlite3.connect(file_name))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.connection.commit()
        self.connection.close()

    def insert_netcdf(self, path):
        for cube in iris.load(path):
            for coord in cube.coords():
                self.insert_coordinate(
                    path,
                    cube.var_name,
                    coord.name())

    def insert_pressure(self, path, variable, values):
        self.cursor.executemany("""
            INSERT OR IGNORE INTO pressure (i, value)
            VALUES(:i, :value)
        """, [dict(i=i, value=value) for i, value in enumerate(values)])

    def pressures(self, path, variable):
        self.cursor.execute("""
            SELECT value FROM pressure
        """)
        rows = self.cursor.fetchall()
        return [p for p, in rows]

    def insert_coordinate(
            self,
            path,
            variable,
            name,
            axis=None):
        self.insert_variable(path, variable)
        self.cursor.execute("""
            INSERT INTO coordinate (name, axis, variable_id)
            VALUES(
                :name,
                :axis,
                (SELECT v.id
                   FROM variable AS v
                   JOIN file AS f
                     ON v.file_id = f.id
                  WHERE f.name = :path
                    AND v.name = :variable))
        """, dict(
            path=path,
            variable=variable,
            name=name,
            axis=axis))

    def coordinates(self, path, variable):
        self.cursor.execute("""
            SELECT c.name
              FROM coordinate AS c
              JOIN variable AS v
                ON c.variable_id = v.id
              JOIN file AS f
                ON v.file_id = f.id
             WHERE f.name = :path
               AND v.name = :variable
        """, dict(path=path, variable=variable))
        rows = self.cursor.fetchall()
        return [name for name, in rows]

    def axis(self, path, variable, coordinate):
        self.cursor.execute("""
            SELECT c.axis
              FROM coordinate AS c
              JOIN variable AS v
                ON c.variable_id = v.id
              JOIN file AS f
                ON v.file_id = f.id
             WHERE f.name = :path
               AND v.name = :variable
               AND c.name = :coordinate
        """, dict(
            path=path,
            variable=variable,
            coordinate=coordinate))
        rows = self.cursor.fetchall()
        return [name for name, in rows]

    def insert_file_name(self, name, initial_time=None):
        self.cursor.execute("""
            INSERT OR IGNORE INTO file (name, initial_time)
            VALUES(:name, :initial_time)
        """, dict(
            name=name,
            initial_time=initial_time))

    def insert_variable(self, path, variable):
        self.insert_file_name(path)
        self.cursor.execute("""
            INSERT INTO variable (name, file_id)
            VALUES(:variable, (SELECT id FROM file WHERE name = :path))
        """, dict(path=path, variable=variable))

    def file_names(self, initial_time=None):
        if initial_time is None:
            self.cursor.execute("""
                SELECT name FROM file
            """)
        else:
            self.cursor.execute("""
                SELECT name FROM file
                 WHERE initial_time = :initial_time
            """, dict(initial_time=initial_time))
        rows = self.cursor.fetchall()
        return [name for name, in rows]

    def variables(self, pattern=None):
        if pattern is None:
            self.cursor.execute("""
                SELECT name FROM variable
            """)
        else:
            self.cursor.execute("""
                SELECT variable.name
                  FROM variable
                  JOIN file
                    ON variable.file_id = file.id
                 WHERE file.name GLOB :pattern
            """, dict(pattern=pattern))
        rows = self.cursor.fetchall()
        return [name for name, in rows]
