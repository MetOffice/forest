"""Minimal Forest implementation"""
import bokeh.plotting


class App(object):
    def __init__(self, title):
        self.title = title

    def __call__(self, document):
        figure = bokeh.plotting.figure(
            title=self.title
        )
        figure.circle([1, 2, 3], [1, 2, 3])
        document.add_root(figure)
