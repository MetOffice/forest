"""Colorbar sub-figure component"""
import bokeh.plotting
from forest.colors import colorbar_figure


class ColorbarUI:
    """Helper to make a figure containing only one colorbar"""
    def __init__(self, color_mapper):
        n = 2
        self.figures = []
        for _ in range(n):
            figure = colorbar_figure(color_mapper, plot_width=200)
            self.figures.append(figure)
        self.layout = bokeh.layouts.column(*self.figures, name="colorbar")
