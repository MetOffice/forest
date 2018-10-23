"""Custom Tornado server to run bokeh apps"""
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
            (r'/static/(.*)', StaticFileHandler, {'path': '_static'}),
            (r'/css/(.*)', StaticFileHandler, {'path': '_static/css'}),
            (r'/images/(.*)', StaticFileHandler, {'path': '_static/images'})
        ])
server.start()

if __name__ == '__main__':
    from bokeh.util.browser import view
    url = 'http://localhost:5006'
    print('Opening Tornado app: {}'.format(url))
    server.io_loop.add_callback(view, url)
    server.io_loop.start()
