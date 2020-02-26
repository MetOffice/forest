"""Colorbar sub-figure component"""
import bokeh.plotting


class ColorbarUI:
    """Helper to make a figure containing only one colorbar"""
    def __init__(self, color_mapper, n=1, name=None):
        # Dimensions
        padding = 5
        margin = 20
        colorbar_height = 20
        plot_height = colorbar_height + 30

        self.figures = []
        self.colorbars = []
        for _ in range(n):
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

        self.layout = bokeh.layouts.row(
                *self.figures,
                sizing_mode="stretch_width",
                name=name)
