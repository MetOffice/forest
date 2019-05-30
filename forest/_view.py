import bokeh.models
from collections import namedtuple


Image = namedtuple("Image", ("path", "variable", "pts"))


class View(object):
    def __init__(self, text, locator=None):
        self.div = bokeh.models.Div(text=text)
        self.locator = locator

    def on_state(self, state):
        image = Image(path=None, variable=state.variable, pts=None)
        if (
                (state.pattern is not None) and
                (state.variable is not None) and
                (state.initial_time is not None) and
                (state.valid_time is not None) and
                (state.pressure is not None)):
            p = float(state.pressure.replace("hPa", ""))
            print(p)
            path, pts = self.locator.path_points(
                state.pattern,
                state.variable,
                state.initial_time,
                state.valid_time,
                p)
            image = Image(path=path, variable=state.variable, pts=pts)
        self.div.text = "<ul><li>{}</li><li>{}</li></ul>".format(
            str(state),
            str(image))
