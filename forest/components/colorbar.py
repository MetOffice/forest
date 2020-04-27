"""Colorbar sub-figure component"""
import bokeh.plotting
from forest.colors import colorbar_figure


class ColorbarUI:
    """Helper to make a figure containing only one colorbar"""
    def __init__(self, color_mapper):
        self.figure = colorbar_figure(color_mapper, plot_width=500)
        self.layout = self.figure
