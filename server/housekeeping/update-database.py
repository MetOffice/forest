#!/usr/bin/env python
"""
Housekeeping task to keep SQL database up-to-date
"""
import argparse
import os
import sqlite3
import glob
import forest.db


def parse_args():
    """Command line parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("bucket_dir")
    parser.add_argument("database_file")
    return parser.parse_args()


def main():
    """Update database with files not already in the database

    - Query database to find existing names
    - List S3 buckets mounted on disk to find addtional files
    - Add missing file entries into the database
    """
    args = parse_args()

    # Search patterns in SQL and on disk
    sql_patterns = [
        "*ga7*.nc",
        "*ga6*.nc",
        "*4p4km*.nc",
        "*philippines*.nc",
        "*malaysia*.nc",
        "*indonesia*.nc",
        "*vietnam*.nc",
    ]

    # SQL database contents
    connection = sqlite3.connect(args.database_file)
    cursor = connection.cursor()
    query = "SELECT name FROM file WHERE name GLOB ?;"
    sql_names = []
    for sql_pattern in sql_patterns:
        rows = cursor.execute(query, (sql_pattern,)).fetchall()
        sql_names += [os.path.basename(row[0]) for row in rows]
    connection.close()

    # S3 bucket contents
    s3_names = []
    for sql_pattern in sql_patterns:
        full_pattern = os.path.join(args.bucket_dir, "wcssp", sql_pattern)
        paths = glob.glob(full_pattern)
        s3_names += [os.path.basename(path) for path in paths]

    # Find extra files
    extra_names = set(s3_names) - set(sql_names)
    extra_paths = [os.path.join(args.bucket_dir, "wcssp", name)
                   for name in extra_names]

    # Add NetCDF files to database
    print("connecting to: {}".format(args.database_file))
    with forest.db.Database.connect(args.database_file) as database:
        for path in extra_paths:
            print("inserting: '{}'".format(path))
            database.insert_netcdf(path)
    print("finished")


if __name__ == '__main__':
    main()
