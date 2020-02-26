"""Colorbar sub-figure component"""
import bokeh.plotting


class ColorbarUI:
    """Helper to make a figure containing only one colorbar"""
    def __init__(self, color_mapper, n=1, name=None):
        self.color_mapper = color_mapper
        self._max_width = 500

        self.figures = []
        self.colorbars = []
        self.color_mappers = []
        self._select = bokeh.models.Select(options=["1", "2", "3", "4"])
        self._select.on_change("value", self._on_select)

        self.row = bokeh.layouts.row(
                *self.figures)
        self.layout = bokeh.layouts.column(
                self._select,
                self.row,
                name=name)

    def _on_select(self, attr, old, new):
        n = int(new)
        print(n)
        self.render(n)

    def render(self, n):
        # Dimensions
        padding = 5
        margin = 20
        colorbar_height = 20
        plot_height = colorbar_height + 30
        if n > len(self.colorbars):
            for i in range(n):
                if len(self.color_mappers) == 0:
                    color_mapper = self.color_mapper
                else:
                    # Color mapper
                    color_mapper = bokeh.models.LinearColorMapper(
                            low=0,
                            high=1,
                            palette=bokeh.palettes.Plasma[256])
                self.color_mappers.append(color_mapper)

                # Colorbar
                colorbar = bokeh.models.ColorBar(
                    color_mapper=color_mapper,
                    location=(0, 0),
                    height=colorbar_height,
                    padding=padding,
                    orientation="horizontal",
                    major_tick_line_color="black",
                    bar_line_color="black",
                    background_fill_alpha=0.,
                )
                colorbar.title = ""
                self.colorbars.append(colorbar)

                # Figure
                figure = bokeh.plotting.figure(
                    plot_height=plot_height,
                    toolbar_location=None,
                    min_border=0,
                    background_fill_alpha=0,
                    border_fill_alpha=0,
                    outline_line_color=None,
                )
                figure.axis.visible = False
                figure.add_layout(colorbar, 'center')
                self.figures.append(figure)

        # Server-side calculation of fixed width
        width = int(self._max_width / n)
        for i in range(n):
            self.figures[i].plot_width = width
            self.colorbars[i].width = int(0.8 * width)
            self.colorbars[i].location = (0.1 * width, 0)
        self.row.children = self.figures[:n]
        self.row.width = self._max_width
