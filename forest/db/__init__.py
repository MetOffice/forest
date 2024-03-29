import functools
import os


@functools.lru_cache(maxsize=None)
def get_database(database_path):
    from forest.db.database import Database

    if database_path != ":memory:" and not os.path.exists(database_path):
        raise ValueError(f"Database file {database_path!r} must exist")
    return Database.connect(database_path)
