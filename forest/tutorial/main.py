import typer
from pathlib import Path


def main(
    build_dir: Path = typer.Argument(
        ..., help="directory in which to build sample files, e.g. '.'"
    )
):
    """
    Builds sample file(s) needed for the tutorial at:

    https://forest-informaticslab.readthedocs.io

    """
    from forest.tutorial import core

    core.build_all(build_dir=build_dir)
