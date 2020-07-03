import sqlite3
import datetime as dt
import forest.db
import forest.db.health


def test_db_health_check():
    """Database tables to monitor S3 object availability"""
    database = forest.db.Database.connect(":memory:")
    database.insert_file_name("file.nc")
    pattern = "*.nc"
    health_db = forest.db.health.HealthDB(database.connection)
    assert health_db.checked_files(pattern) == ["file.nc"]


def test_db_health_check_mark_oserror():
    """Database tables to monitor S3 object availability"""
    database = forest.db.Database.connect(":memory:")
    database.insert_file_name("file-0.nc")
    health_db = forest.db.health.HealthDB(database.connection)
    health_db.insert_error("file-1.nc",
                           OSError("Error message"),
                           dt.datetime(2020, 1, 1))
    pattern = "*.nc"
    assert health_db.checked_files(pattern) == ["file-0.nc", "file-1.nc"]
