"""
Command line interface to FOREST application
"""
import os
import argparse
import bokeh.command.bootstrap
from forest.parse_args import add_arguments


APP_PATH = os.path.join(os.path.dirname(__file__), "..")


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)

    # FOREST specific arguments
    group = parser.add_argument_group('forest arguments')
    add_arguments(group)

    # Bokeh serve pass-through arguments
    group = parser.add_argument_group('bokeh serve arguments')
    group.add_argument(
        "--dev", action="store_true",
        help="run server in development mode")
    group.add_argument(
        "--port",
        help="port to listen on")
    group.add_argument(
        "--show", action="store_true",
        help="launch browser")
    group.add_argument(
        "--allow-websocket-origin", metavar="HOST[:PORT]",
        help="public hostnames that may connect to the websocket")

    args = parser.parse_args(args=args)
    if len(args.files) == 0 and args.config_file is None:
        parser.error("please specify file(s) or a valid --config-file file")
    return args


def main():
    args = parse_args()
    bokeh.command.bootstrap.main(bokeh_args(APP_PATH, args))


def bokeh_args(app_path, args):
    """translate from forest to bokeh serve command"""
    opts = ["bokeh", "serve", app_path]
    if args.dev:
        opts += ["--dev"]
    if args.show:
        opts += ["--show"]
    if args.port:
        opts += ["--port", str(args.port)]
    if args.allow_websocket_origin:
        opts += ["--allow-websocket-origin", str(args.allow_websocket_origin)]
    extra = []
    if args.config_file is not None:
        extra += ["--config-file", str(args.config_file)]
    if args.database is not None:
        extra += ["--database", str(args.database)]
    if args.directory is not None:
        extra += ["--directory", str(args.directory)]
    if args.file_type != "unified_model":
        extra += ["--file-type", str(args.file_type)]
    if len(args.files) > 0:
        extra += args.files
    if len(extra) > 0:
        opts += ["--args"] + extra
    return opts
