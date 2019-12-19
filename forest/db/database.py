try:
    import iris
except ImportError:
    # ReadTheDocs can't install iris
    pass
import netCDF4
import jinja2
from .connection import Connection


__all__ = [
    "Database",
    "CoordinateDB"
]


class CoordinateDB(Connection):
    def __init__(self, connection):
        self.connection = connection
        self.cursor = self.connection.cursor()
        self.cursor.execute("""
        CREATE TABLE file (
                  id INTEGER PRIMARY KEY,
                name TEXT,
                UNIQUE(name))
        """)
        self.cursor.execute("""
        CREATE TABLE axis (
                  id INTEGER PRIMARY KEY,
            variable TEXT,
                name TEXT,
               value INTEGER,
             file_id INTEGER,
             FOREIGN KEY(file_id) REFERENCES file(id))
        """)
        self.cursor.execute("""
        CREATE TABLE time (
                  id INTEGER PRIMARY KEY,
            variable TEXT,
                   i INTEGER,
               value TEXT,
             file_id INTEGER,
             FOREIGN KEY(file_id) REFERENCES file(id))
        """)
        self.cursor.execute("""
        CREATE TABLE pressure (
                  id INTEGER PRIMARY KEY,
            variable TEXT,
                   i INTEGER,
               value REAL,
             file_id INTEGER,
             FOREIGN KEY(file_id) REFERENCES file(id))
        """)

    def insert_pressures(self, path, variable, values):
        self.cursor.execute("""
            INSERT OR IGNORE INTO file (name) VALUES (:path)
        """, dict(path=path))
        query = """
        INSERT INTO pressure (variable, i, value, file_id)
             VALUES (
                 :variable,
                 :i,
                 :value,
                 (SELECT id FROM file WHERE name = :path))
        """
        data = [dict(
            path=path,
            variable=variable,
            i=i,
            value=value) for i, value in enumerate(values)]
        self.cursor.executemany(query, data)

    def pressure_index(self, pattern, variable, value):
        self.cursor.execute("""
        SELECT pressure.i
          FROM pressure
          JOIN file
            ON file.id = pressure.file_id
         WHERE file.name GLOB :pattern
           AND pressure.variable = :variable
           AND pressure.value = :value
        """, dict(
            pattern=pattern,
            variable=variable,
            value=value))
        rows = self.cursor.fetchall()
        return [i for i, in rows]

    def insert_times(self, path, variable, values):
        self.cursor.execute("""
            INSERT OR IGNORE INTO file (name) VALUES (:path)
        """, dict(path=path))
        data = [dict(
            path=path,
            variable=variable,
            i=i,
            value=value) for i, value in enumerate(values)]
        self.cursor.executemany("""
            INSERT INTO time (variable, i, value, file_id)
                 VALUES (
                 :variable,
                 :i,
                 :value,
                 (SELECT id FROM file WHERE name = :path))
        """, data)

    def time_index(self, pattern, variable, value):
        self.cursor.execute("""
        SELECT time.i
          FROM time
          JOIN file
            ON file.id = time.file_id
         WHERE file.name GLOB :pattern
           AND time.variable = :variable
           AND time.value = :value
        """, dict(
            pattern=pattern,
            variable=variable,
            value=value))
        rows = self.cursor.fetchall()
        return [i for i, in rows]

    def insert_axis(self, path, variable, coordinate, axis):
        self.cursor.execute("""
        INSERT OR IGNORE INTO file (name) VALUES (:path)
        """, dict(path=path))
        self.cursor.execute("""
        INSERT INTO axis (variable, name, value, file_id)
             VALUES (
                     :variable,
                     :coordinate,
                     :axis,
                     (SELECT id FROM file WHERE name = :path))
        """, dict(
            path=path,
            variable=variable,
            coordinate=coordinate,
            axis=axis))

    def axis(self, path, variable, coordinate):
        self.cursor.execute("""
        SELECT value FROM axis
          JOIN file
            ON file.id = axis.file_id
         WHERE file.name = :path
           AND axis.variable = :variable
           AND axis.name = :coordinate
        """, dict(
            path=path,
            variable=variable,
            coordinate=coordinate))
        rows = self.cursor.fetchall()
        return rows[0][0]

    def coordinates(self, path, variable):
        self.cursor.execute("""
        SELECT axis.name, axis.value FROM axis
          JOIN file
            ON file.id = axis.file_id
         WHERE file.name = :path
           AND axis.variable = :variable
        """, dict(
            path=path,
            variable=variable))
        return self.cursor.fetchall()




class Database(Connection):
    """Stores index and paths of forecast diagnostics"""
    def __init__(self, connection):
        self.connection = connection
        self.cursor = self.connection.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS file (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    reference TEXT,
                    UNIQUE(name))
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS variable (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    time_axis INTEGER,
                    pressure_axis INTEGER,
                    file_id INTEGER,
                    FOREIGN KEY(file_id) REFERENCES file(id),
                    UNIQUE(name, file_id))
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS pressure (
                    id INTEGER PRIMARY KEY,
                    i INTEGER,
                    value REAL,
                    UNIQUE(i, value)
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS variable_to_pressure (
                    variable_id INTEGER,
                    pressure_id INTEGER,
                    PRIMARY KEY(variable_id, pressure_id),
                    FOREIGN KEY(variable_id) REFERENCES variable(id),
                    FOREIGN KEY(pressure_id) REFERENCES pressure(id))
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS time (
                    id INTEGER PRIMARY KEY,
                    i INTEGER,
                    value TEXT,
                    UNIQUE(i, value))
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS variable_to_time (
                    variable_id INTEGER,
                    time_id INTEGER,
                    PRIMARY KEY(variable_id, time_id),
                    FOREIGN KEY(variable_id) REFERENCES variable(id),
                    FOREIGN KEY(time_id) REFERENCES time(id))
        """)

    def insert_netcdf(self, path):
        """Coordinate and meta-data information taken from NetCDF file"""
        with netCDF4.Dataset(path) as dataset:
            try:
                obj = dataset.variables["forecast_reference_time"]
                reference_time = netCDF4.num2date(obj[:], units=obj.units)
            except KeyError:
                reference_time = None

        self.insert_file_name(path, reference_time=reference_time)

        cubes = iris.load(path)
        for cube in cubes:
            variable = cube.var_name
            time_axis = self._axis(cube, 'time')
            pressure_axis = self._axis(cube, 'pressure')
            self.insert_variable(
                path,
                variable,
                time_axis=time_axis,
                pressure_axis=pressure_axis)
            try:
                times = [cell.point for cell in cube.coord('time').cells()]
                self.insert_times(path, variable, times)
            except iris.exceptions.CoordinateNotFoundError:
                pass
            try:
                pressures = [cell.point for cell in cube.coord('pressure').cells()]
                self.insert_pressures(path, variable, pressures)
            except iris.exceptions.CoordinateNotFoundError:
                pass

    @staticmethod
    def _axis(cube, coord):
        try:
            dims = cube.coord_dims(coord)
            if len(dims) == 0:
                return None
            else:
                return dims[0]
        except iris.exceptions.CoordinateNotFoundError:
            return None

    def initial_times(self, pattern=None, variable=None):
        """Distinct initialisation times"""
        if pattern is None:
            query = """
                SELECT DISTINCT reference
                  FROM file
                 WHERE reference IS NOT NULL
                 ORDER BY reference
            """
        else:
            query = """
                SELECT DISTINCT reference
                  FROM file
                 WHERE reference IS NOT NULL
                   AND name GLOB :pattern
                 ORDER BY reference;
            """
        self.cursor.execute(query, dict(pattern=pattern))
        rows = self.cursor.fetchall()
        return [r for r, in rows]

    def files(self, pattern=None):
        """File names"""
        if pattern is None:
            query = """
                SELECT name
                  FROM file
                 ORDER BY name;
            """
        else:
            query = """
                SELECT name
                  FROM file
                 WHERE name GLOB :pattern
                 ORDER BY name;
            """
        self.cursor.execute(query, dict(pattern=pattern))
        rows = self.cursor.fetchall()
        return [r for r, in rows]

    def variables(self, pattern=None):
        # Note: SQL injection possible if not properly escaped
        #       use ? and :name syntax in template
        environment = jinja2.Environment(extensions=['jinja2.ext.do'])
        query = environment.from_string("""
            SELECT DISTINCT variable.name
              FROM variable
              {% if pattern is not none %}
              JOIN file
                ON file.id = variable.file_id
             WHERE file.name GLOB :pattern
              {% endif %}
             ORDER BY variable.name;
        """).render(pattern=pattern)
        self.cursor.execute(query, dict(pattern=pattern))
        rows = self.cursor.fetchall()
        return [r for r, in rows]

    def insert_file_name(self, path, reference_time=None):
        if reference_time is not None:
            reference_time = str(reference_time)
        self.cursor.execute("""
            INSERT OR IGNORE INTO file (name, reference)
            VALUES (:path, :reference)
        """, dict(path=path, reference=reference_time))

    def insert_variable(
            self,
            path,
            variable,
            time_axis=None,
            pressure_axis=None):
        self.insert_file_name(path)
        self.cursor.execute("""
            INSERT OR IGNORE
                        INTO variable (name, time_axis, pressure_axis, file_id)
                      VALUES (
                             :variable,
                             :time_axis,
                             :pressure_axis,
                             (SELECT id FROM file WHERE name=:path))
        """, dict(
            path=path,
            variable=variable,
            time_axis=time_axis,
            pressure_axis=pressure_axis))

    def insert_pressures(self, path, variable, values):
        """Helper method to insert a coordinate related to a variable"""
        for i, value in enumerate(values):
            self.insert_pressure(path, variable, value, i)

    def insert_pressure(self, path, variable, pressure, i):
        self.insert_variable(path, variable)
        self.cursor.execute("""
            INSERT OR IGNORE INTO pressure (i, value) VALUES (:i,:pressure)
        """, dict(i=i, pressure=pressure))
        self.cursor.execute("""
            INSERT OR IGNORE INTO variable_to_pressure (variable_id, pressure_id)
            VALUES(
                (SELECT variable.id FROM variable
                   JOIN file ON variable.file_id = file.id
                  WHERE file.name = :path AND variable.name=:variable),
                (SELECT id FROM pressure WHERE value=:pressure AND i=:i))
        """, dict(path=path, variable=variable, pressure=pressure, i=i))

    def valid_times(self,
                    pattern=None,
                    variable=None,
                    initial_time=None):
        """Valid times associated with search criteria"""
        # Note: SQL injection possible if not properly escaped
        #       use ? and :name syntax in template
        environment = jinja2.Environment(extensions=['jinja2.ext.do'])
        query = environment.from_string("""
            {% set EQNS = [] %}
            {% if initial_time is not none %}
               {% do EQNS.append('file.reference = :initial_time') %}
            {% endif %}
            {% if pattern is not none %}
               {% do EQNS.append('file.name GLOB :pattern') %}
            {% endif %}
            {% if variable is not none %}
               {% do EQNS.append('v.name = :variable') %}
            {% endif %}
            SELECT time.value
              FROM time
             {% if EQNS %}
              JOIN variable_to_time AS vt
                ON vt.time_id = time.id
              JOIN variable AS v
                ON vt.variable_id = v.id
              JOIN file
                ON v.file_id = file.id
             WHERE {{ EQNS | join(' AND ') }}
             {% endif %}
        """).render(
            initial_time=initial_time,
            variable=variable,
            pattern=pattern)
        self.cursor.execute(query, dict(
            variable=variable,
            pattern=pattern,
            initial_time=initial_time))
        rows = self.cursor.fetchall()
        return [time for time, in rows]

    def pressures(self, pattern=None, variable=None, initial_time=None):
        """Select pressures from database"""
        # Note: SQL injection possible if not properly escaped
        #       use ? and :name syntax in template
        environment = jinja2.Environment(extensions=['jinja2.ext.do'])
        query = environment.from_string("""
            {% set EQNS = [] %}
            {% if variable is not none %}
               {% do EQNS.append('v.name = :variable') %}
            {% endif %}
            {% if pattern is not none %}
               {% do EQNS.append('file.name GLOB :pattern') %}
            {% endif %}
            {% if initial_time is not none %}
               {% do EQNS.append('file.reference = :initial_time') %}
            {% endif %}
            {% if EQNS %}
            SELECT DISTINCT pressure.value
              FROM pressure
              JOIN variable_to_pressure AS vp
                ON vp.pressure_id = pressure.id
              JOIN variable AS v
                ON v.id = vp.variable_id
              JOIN file
                ON v.file_id = file.id
             WHERE {{ EQNS | join(' AND ') }}
             ORDER BY value
             {% else %}
            SELECT DISTINCT value
              FROM pressure
             ORDER BY value
             {% endif %}
        """).render(
            variable=variable,
            pattern=pattern,
            initial_time=initial_time)
        self.cursor.execute(query, dict(
            variable=variable,
            pattern=pattern,
            initial_time=initial_time))
        rows = self.cursor.fetchall()
        return [time for time, in rows]

    def fetch_times(self, path, variable):
        """Helper method to find times related to a variable"""
        self.cursor.execute("""
            SELECT value FROM time
        """)
        return [time for time, in self.cursor.fetchall()]

    def insert_times(self, path, variable, times):
        """Helper method to insert a time coordinate related to a variable"""
        for i, time in enumerate(times):
            self.insert_time(path, variable, time, i)

    def insert_time(self, path, variable, time, i):
        time = str(time)
        self.insert_variable(path, variable)
        self.cursor.execute("""
            INSERT OR IGNORE INTO time (i, value) VALUES (:i,:value)
        """, dict(i=i, value=time))
        self.cursor.execute("""
            INSERT OR IGNORE INTO variable_to_time (variable_id, time_id)
            VALUES(
                (SELECT variable.id FROM variable
                   JOIN file ON variable.file_id = file.id
                  WHERE file.name=:path AND variable.name=:variable),
                (SELECT id FROM time WHERE value=:value AND i=:i))
        """, dict(path=path, variable=variable, value=time, i=i))

    def find_time(self, variable, time):
        self.cursor.execute("""
            SELECT file.name, time.i FROM file
              JOIN variable ON file.id = variable.file_id
              JOIN variable_to_time AS junction ON variable.id = junction.variable_id
              JOIN time ON time.id = junction.time_id
             WHERE variable.name = :variable AND time.value = :time
        """, dict(variable=variable, time=time))
        return self.cursor.fetchall()

    def find_pressure(self, variable, pressure):
        return self.find(variable, pressure)

    def find(self, variable, pressure):
        self.cursor.execute("""
            SELECT file.name, pressure.i FROM file
              JOIN variable ON file.id = variable.file_id
              JOIN variable_to_pressure AS junction ON variable.id = junction.variable_id
              JOIN pressure ON pressure.id = junction.pressure_id
            WHERE variable.name = :variable AND pressure.value = :pressure
        """, dict(variable=variable, pressure=pressure))
        return self.cursor.fetchall()

    def file_names(self):
        self.cursor.execute("SELECT name FROM file")
        return [row[0] for row in self.cursor.fetchall()]

    def fetch_dates(self, pattern=None):
        self.cursor.execute("""
            SELECT DISTINCT value FROM time
        """)
        rows = self.cursor.fetchall()
        return [row[0] for row in rows]
