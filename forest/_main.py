#!/usr/bin/env python3
import argparse
import _database as db


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database", required=True,
        help="database file to write/extend")
    parser.add_argument(
        "paths", nargs="+", metavar="FILE",
        help="unified model netcdf files")
    return parser.parse_args(args=argv)


def main(argv=None):
    args = parse_args(argv=argv)
    with db.Database.connect(args.database) as database:
        for path in args.paths:
            print("reading: {}".format(path))
            database.insert_netcdf(path)


if __name__ == '__main__':
    main()
