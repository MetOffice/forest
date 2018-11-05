#!/usr/bin/env python
"""Custom Tornado server to run bokeh apps"""
import argparse
import os
from jinja2 import Environment, FileSystemLoader
from tornado.web import RequestHandler, StaticFileHandler
from bokeh.server.server import Server

import app.main

script_dir = os.path.dirname(__file__)
env = Environment(loader=FileSystemLoader(script_dir))


class IndexHandler(RequestHandler):
    def get(self):
        template = env.get_template("index.html")
        self.write(template.render())


def full_path(file_name):
    return os.path.join(script_dir, file_name)


def bokeh_server(**kwargs):
    highway = app.main.load_app(full_path("highway.yaml"))
    wcssp_south_east_asia = app.main.load_app(full_path("wcssp_south_east_asia.yaml"))
    routes = {
        '/highway': highway,
        '/wcssp_south_east_asia': wcssp_south_east_asia
    }
    extra_patterns = [
        (r'/', IndexHandler),
        (r'/_static/(.*)', StaticFileHandler, {'path': full_path('_static')}),
        (r'/app/static/(.*)', StaticFileHandler, {'path': full_path('app/static')}),
        (r'/app/static/css/(.*)', StaticFileHandler, {'path': full_path('app/static/css')}),
    ]
    return Server(routes,
                  num_procs=1,
                  extra_patterns=extra_patterns,
                  **kwargs)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show", action="store_true",
        help="launch browser on startup")
    parser.add_argument(
        "--port", default=5006, type=int,
        help="port to listen on")
    parser.add_argument(
        "--allow-websocket-origin",
        metavar="HOST[:PORT]", action="append",
        help=("public hostnames which may connect "
              "to the bokeh websocket"))
    parser.add_argument(
        "--unused-session-lifetime",
        default=15000,
        metavar="MILLISECONDS",
        help="how long unused sessions last")
    return parser.parse_args(args=argv)


def main():
    args = parse_args()
    url = 'http://localhost:{}'.format(args.port)
    print('Opening Tornado app: {}'.format(url))
    server = bokeh_server(
        port=args.port,
        allow_websocket_origin=args.allow_websocket_origin,
        unused_session_lifetime=args.unused_session_lifetime
    )
    if args.show:
        from bokeh.util.browser import view
        server.io_loop.add_callback(view, url)
    server.io_loop.start()


if __name__ == '__main__':
    main()
