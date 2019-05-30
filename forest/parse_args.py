import argparse
import yaml


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
    return parser.parse_args(args=argv)


class Namespace(object):
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


def load_config(stream):
    data = yaml.load(stream)
    patterns = [(m["name"], m["pattern"]) for m in data["models"]]
    return Namespace(patterns=patterns)
