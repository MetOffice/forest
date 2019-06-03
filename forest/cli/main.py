import argparse
import forest.db.main


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    database_parser = subparsers.add_parser("database")
    forest.db.main.add_arguments(database_parser)
    database_parser.set_defaults(main=forest.db.main.main)

    serve_parser = subparsers.add_parser("serve")
    # TODO: delegate to serve command
    args = parser.parse_args()
    if args.main is not None:
        args.main(args=args)
