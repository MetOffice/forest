"""Capture feedback related to displayed data"""
import bokeh.layouts
import bokeh.models


class Survey(object):
    def __init__(self):
        self.div = bokeh.models.Div(text="Survey")
        self.layout = bokeh.layouts.column(self.div)
