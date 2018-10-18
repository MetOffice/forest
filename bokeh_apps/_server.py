"""Custom Tornado server to run bokeh apps"""
from jinja2 import Environment, FileSystemLoader
import tornado.web
from tornado.web import RequestHandler
from bokeh.embed import server_document
from bokeh.server.server import Server
import bokeh.plotting

import highway.main
import wcssp.main

env = Environment(loader=FileSystemLoader('_templates'))

class IndexHandler(RequestHandler):
    def get(self):
        template = env.get_template("app_index.html")
        apps = [
                {"name": "WCSSP", "href": "wcssp"},
                {"name": "HIGHWAY", "href": "highway"}
        ]
        self.write(template.render(apps=apps))

server = Server(
        {'/highway': highway.main.app,
         '/wcssp': wcssp.main.app},
        num_procs=1,
        extra_patterns=[
            (r'/', IndexHandler),
            (r'/static/(.*)', tornado.web.StaticFileHandler, {'path': '_static'})
        ])
server.start()

if __name__ == '__main__':
    from bokeh.util.browser import view
    print('Opening Tornado app')
    server.io_loop.add_callback(view, 'http://localhost:5006')
    server.io_loop.start()
