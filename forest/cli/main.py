import os
import argparse
import subprocess
import forest.db.main


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    database_parser = subparsers.add_parser("database")
    forest.db.main.add_arguments(database_parser)
    database_parser.set_defaults(main=forest.db.main.main)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.set_defaults(main=serve_main)

    args = parser.parse_args()
    if args.main is not None:
        args.main(args=args)


def serve_main(args=None):
    """Entry-point for forest serve command"""
    subprocess.call([
        "bokeh",
        "serve",
        os.path.join(os.path.dirname(__file__), ".."),
        "--args",
        "--database", "database.db",
        "--config-file", "config.yml"
    ])
