"""
S3 object health status
"""
import sqlite3


class HealthDB:
    """Maintain meta-data related to S3 objects"""
    def __init__(self, connection):
        self.connection = connection
        self.cursor = self.connection.cursor()
        self.cursor.execute("""
            CREATE TABLE
           IF NOT EXISTS health (
                      id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                   errno INTEGER,
                strerror TEXT,
                    time TEXT,
                         UNIQUE(name))
        """)

    @classmethod
    def connect(cls, path_or_memory):
        """Connect to sqlite3 database"""
        return cls(sqlite3.connect(path_or_memory))

    def checked_files(self, pattern):
        """Files that are in the database

        :returns files: either successfully processed or marked as OSError
        """
        return sorted(set(self.files(pattern)) |
                      set(self.error_files(pattern)))

    def files(self, pattern):
        query = "SELECT name FROM file WHERE name GLOB :pattern;"
        params = {"pattern": pattern}
        return [path for path, in self.cursor.execute(query, params)]

    def error_files(self, pattern):
        query = "SELECT name FROM health WHERE name GLOB :pattern;"
        params = {"pattern": pattern}
        return [path for path, in self.cursor.execute(query, params)]

    def insert_error(self, path, error, check_time):
        """Insert OSError into table"""
        query = """
            INSERT OR IGNORE
              INTO health (name, errno, strerror, time)
            VALUES (:path, :errno, :strerror, :time);
        """
        params = {
            "path": path,
            "errno": error.errno,
            "strerror": error.strerror,
            "time": check_time.isoformat()
        }
        self.cursor.execute(query, params)
