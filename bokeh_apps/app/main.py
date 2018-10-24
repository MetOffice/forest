"""Minimal Forest implementation"""
import yaml
import bokeh.plotting


def load_app(config_file):
    with open(config_file) as stream:
        settings = yaml.load(stream)
    return App(settings["title"])


class App(object):
    def __init__(self, title):
        self.title = title

    def __call__(self, document):
        print("called: {} {}".format(self.title, document))
        figure = bokeh.plotting.figure(
            title=self.title
        )
        figure.circle([1, 2, 3], [1, 2, 3])
        document.add_root(figure)
