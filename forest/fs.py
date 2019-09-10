import bokeh.layouts
from observe import Observer


class Controls(Observer):
    def __init__(self):
        self.layout = bokeh.layouts.column()
        super().__init__()
