"""
Command line interface to FOREST application
"""
import os
import argparse
import bokeh.command.bootstrap
from forest.parse_args import add_arguments


APP_PATH = os.path.join(os.path.dirname(__file__), "..")


def main(argv=None):
    bokeh.command.bootstrap.main(bokeh_command(APP_PATH, argv=argv))


def bokeh_command(app_path, argv):
    """Translate from forest to bokeh serve command"""
    # Note: forest_opts is a list of command line flags
    bokeh_args, forest_opts = parse_args(args=argv)
    opts = ["bokeh", "serve", app_path]
    if bokeh_args.dev:
        opts += ["--dev"]
    if bokeh_args.show:
        opts += ["--show"]
    if bokeh_args.port:
        opts += ["--port", str(bokeh_args.port)]
    if bokeh_args.allow_websocket_origin:
        opts += ["--allow-websocket-origin", str(bokeh_args.allow_websocket_origin)]
    if len(forest_opts) > 0:
        opts += ["--args"] + forest_opts
    return opts


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)

    # FOREST specific arguments
    group = parser.add_argument_group('forest arguments')
    add_arguments(group)

    # Bokeh serve pass-through arguments
    group = parser.add_argument_group('bokeh serve arguments')
    add_bokeh_arguments(group)

    # Only parse bokeh serve args do not touch forest.main args
    _parser = argparse.ArgumentParser(add_help=False)
    add_bokeh_arguments(_parser)
    bk_args, extra = _parser.parse_known_args(args=args)

    args = parser.parse_args(args=args)
    if len(args.files) == 0 and args.config_file is None:
        parser.error("please specify file(s) or a valid --config-file file")
    return bk_args, extra


def add_bokeh_arguments(parser):
    parser.add_argument(
        "--dev", action="store_true",
        help="run server in development mode")
    parser.add_argument(
        "--port",
        help="port to listen on")
    parser.add_argument(
        "--show", action="store_true",
        help="launch browser")
    parser.add_argument(
        "--allow-websocket-origin", metavar="HOST[:PORT]",
        help="public hostnames that may connect to the websocket")

            
