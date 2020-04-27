"""Colorbar sub-figure component"""
import bokeh.plotting
from forest.colors import colorbar_figure, parse_color_spec


class ColorbarUI:
    """Helper to make a figure containing only one colorbar"""
    def __init__(self):
        n = 1
        self.figures = []
        self.color_mappers = []
        for _ in range(n):
            figure, color_mapper = self.make_colorbar()
            self.color_mappers.append(color_mapper)
            self.figures.append(figure)
        self.layout = bokeh.layouts.column(*self.figures, name="colorbar")

    def connect(self, store):
        store.add_subscriber(self.render)

    def render(self, state):
        """Query state for color_mapper settings"""
        layers = state.get("layers", {}).get("index", {})
        specs = []
        for _, settings in sorted(layers.items()):
            if "colorbar" in settings:
                spec = parse_color_spec(settings["colorbar"])
                specs.append(spec)

        # Balance number of color_mappers
        missing = len(specs) - len(self.color_mappers)
        if missing > 0:
            for _ in range(missing):
                figure, color_mapper = self.make_colorbar()
                self.color_mappers.append(color_mapper)
                self.figures.append(figure)

        # Add/reuse figures and color_mappers
        for i, spec in enumerate(specs):
            spec.apply(self.color_mappers[i])

        # Update layout
        self.layout.children = self.figures[:len(specs)]

    def make_colorbar(self):
        color_mapper = bokeh.models.LinearColorMapper(
            palette="Greys256",
            low=0,
            high=1)
        figure = colorbar_figure(color_mapper, plot_width=200)
        return figure, color_mapper
