#!/usr/bin/env python3
import argparse
from . import database as db


def parse_args(argv=None, parser=None):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(args=argv)


def add_arguments(parser):
    parser.add_argument(
        "--database", required=True,
        help="database file to write/extend")
    parser.add_argument(
        "paths", nargs="+", metavar="FILE",
        help="unified model netcdf files")


def main(argv=None, args=None):
    if args is None:
        args = parse_args(argv=argv)
    with db.Database.connect(args.database) as database:
        for path in args.paths:
            database.insert_netcdf(path)


if __name__ == '__main__':
    main()
