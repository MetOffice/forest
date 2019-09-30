"""Command-line interface to forest-tutorial

This script builds the sample file(s) needed to
follow along with the tutorial at:

https://forest-informaticslab.readthedocs.io

"""
import argparse
import os
from . import core


class HelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawTextHelpFormatter):
    pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=HelpFormatter)
    parser.add_argument("--build-dir",
            metavar="DIR",
            default=os.getcwd(),
            help="directory in which to build sample files")
    return parser.parse_args(args=argv)


def main(argv=None):
    args = parse_args(argv=argv)
    core.build_all(build_dir=args.build_dir)
