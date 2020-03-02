"""Colorbar sub-figure component"""
import bokeh.plotting


class ColorbarUI:
    """Helper to make a figure containing only one colorbar"""
    def __init__(self, color_mapper):
        # Dimensions
        padding = 5
        margin = 20
        colorbar_height = 20
        plot_height = colorbar_height + 30
        plot_width = 500

        # Colorbar
        self.colorbar = bokeh.models.ColorBar(
            color_mapper=color_mapper,
            location=(0, 0),
            height=colorbar_height,
            width=int(plot_width - (margin + padding)),
            padding=padding,
            orientation="horizontal",
            major_tick_line_color="black",
            bar_line_color="black",
            background_fill_alpha=0.,
        )
        self.colorbar.title = ""

        # Figure
        self.figure = bokeh.plotting.figure(
            plot_height=plot_height,
            plot_width=plot_width,
            toolbar_location=None,
            min_border=0,
            background_fill_alpha=0,
            border_fill_alpha=0,
            outline_line_color=None,
        )
        self.figure.axis.visible = False
        self.figure.add_layout(self.colorbar, 'center')

        self.layout = self.figure
