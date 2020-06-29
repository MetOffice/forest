"""A very basic title component"""
import datetime as dt
import bokeh.models
import bokeh.layouts
import forest.state


class Title:
    """Simple title"""
    def __init__(self):
        self.div = bokeh.models.Div(text="")
        self.layout = bokeh.layouts.row(self.div, name="title")

    def connect(self, store):
        store.add_subscriber(self.render)

    def render(self, state):
        if isinstance(state, dict):
            state = forest.state.State.from_dict(state)
        text = ""
        if isinstance(state.valid_time, dt.datetime):
            text = f"{state.valid_time:%A %d %B %Y %H:%M}"
            if isinstance(state.initial_time, dt.datetime):
                elapsed = state.valid_time - state.initial_time
                hours = int((elapsed).total_seconds() / 3600)
                text = f"{text} T{hours:+}"
        self.div.text = text
