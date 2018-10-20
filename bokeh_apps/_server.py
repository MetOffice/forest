"""Custom Tornado server to run bokeh apps"""
from jinja2 import Environment, FileSystemLoader
from tornado.web import RequestHandler, StaticFileHandler
from bokeh.server.server import Server

import highway.main
import wcssp.main

env = Environment(loader=FileSystemLoader('_templates'))

class IndexHandler(RequestHandler):
    def get(self):
        template = env.get_template("index.html")
        self.write(template.render())

server = Server(
        {'/highway': highway.main.app,
         '/wcssp': wcssp.main.app},
        num_procs=4,
        extra_patterns=[
            (r'/', IndexHandler),
            (r'/static/(.*)', StaticFileHandler, {'path': '_static'}),
            (r'/css/(.*)', StaticFileHandler, {'path': '_static/css'}),
            (r'/images/(.*)', StaticFileHandler, {'path': '_static/images'})
        ])
server.start()

if __name__ == '__main__':
    from bokeh.util.browser import view
    print('Opening Tornado app')
    server.io_loop.add_callback(view, 'http://localhost:5006')
    server.io_loop.start()
