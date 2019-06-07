import sqlite3


class Connection(object):
    def __init__(self, connection):
        self.connection = connection
        self.cursor = self.connection.cursor()

    @classmethod
    def connect(cls, path):
        """Create database instance from location on disk or :memory:"""
        return cls(sqlite3.connect(path))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.connection.commit()
        self.connection.close()
