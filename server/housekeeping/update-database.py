#!/usr/bin/env python
"""
Housekeeping task to keep SQL database up-to-date
"""
import argparse
import os
import sqlite3
import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("bucket_dir")
    parser.add_argument("database_file")
    return parser.parse_args()


def main():
    args = parse_args()

    # SQL database contents
    query = "SELECT name FROM file WHERE name GLOB ?;"
    sql_glob_pattern = "*vietnam*"
    connection = sqlite3.connect(args.database_file)
    cursor = connection.cursor()
    rows = cursor.execute(query, sql_glob_pattern).fetchall()
    sql_names = [os.path.basename(row[0]) for row in rows]
    connection.close()

    # S3 bucket contents
    pattern = os.path.join(args.bucket_dir, "wcssp/*vietnam*.nc")
    paths = glob.glob(pattern)
    s3_names = [os.path.basename(path) for path in paths]

    # Summary
    extra_names = set(s3_names) - set(sql_names)
    for name in sorted(extra_names):
        print(name)


if __name__ == '__main__':
    main()
