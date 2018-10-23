#!/usr/bin/env python
"""Custom Tornado server to run bokeh apps"""
import argparse
import os
from jinja2 import Environment, FileSystemLoader
from tornado.web import RequestHandler, StaticFileHandler
from bokeh.server.server import Server

import highway.main
import wcssp_south_east_asia.main

env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))

class IndexHandler(RequestHandler):
    def get(self):
        template = env.get_template("index.html")
        self.write(template.render())

server = Server(
        {'/highway': highway.main.app,
         '/wcssp_south_east_asia': wcssp_south_east_asia.main.app},
        num_procs=1,
        extra_patterns=[
            (r'/', IndexHandler),
            (r'/static/(.*)', StaticFileHandler, {'path': 'static'}),
            (r'/css/(.*)', StaticFileHandler, {'path': 'static/css'}),
            (r'/images/(.*)', StaticFileHandler, {'path': 'static/images'})
        ],
        port=5006)
server.start()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true",
                        help="launch browser on startup")
    return parser.parse_args(args=argv)


def main():
    args = parse_args()
    url = 'http://localhost:5006'
    print('Opening Tornado app: {}'.format(url))
    if args.show:
        from bokeh.util.browser import view
        server.io_loop.add_callback(view, url)
    server.io_loop.start()


if __name__ == '__main__':
    main()
