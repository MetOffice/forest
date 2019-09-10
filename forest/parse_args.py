import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        help="directory to use with paths returned from database")
    parser.add_argument(
        "--database",
        help="SQL database to optimise menu system")
    parser.add_argument(
        "--config-file",
        metavar="YAML_FILE",
        help="YAML file to configure application")
    parser.add_argument(
        "files", nargs="*", metavar="FILE",
        help="FILE(s) to display")
    args = parser.parse_args(args=argv)
    if (
            (args.config_file is None) and
            (len(args.files) == 0)):
        msg = "Either specify file(s) or --config-file"
        parser.error(msg)
    return args
