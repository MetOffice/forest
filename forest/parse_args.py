import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(args=argv)
    if (
            (args.config_file is None) and
            (len(args.files) == 0)):
        msg = "Either specify file(s) or --config-file"
        parser.error(msg)
    return args


def add_arguments(parser):
    parser.add_argument(
        "files", nargs="*", metavar="FILE",
        help="FILE(s) to display")
    parser.add_argument(
        "--config-file",
        metavar="YAML_FILE",
        help="YAML file to configure application")
    parser.add_argument(
        "--file-type", default="unified_model", metavar="FILETYPE",
        help="keyword to navigate/display file(s)")
    parser.add_argument(
        "--var", action="append", dest="variables",
        nargs=2, metavar=("KEY", "VALUE"),
        help="variable(s) to substitute in --config-file, may be repeated")
    parser.add_argument(
        "--auto-shutdown", action="store_true", default=False,
        help="server shutdown on tab close - for desktop versions.")    
    