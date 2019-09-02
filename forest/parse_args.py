import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        help="directory to use with paths returned from database")
    parser.add_argument(
        "--database",
        required=True,
        help="SQL database to optimise menu system")
    parser.add_argument(
        "--config-file",
        required=True, metavar="YAML_FILE",
        help="YAML file to configure application")
    parser.add_argument(
        "--survey-db",
        help="json file to store survey results")
    return parser.parse_args(args=argv)
