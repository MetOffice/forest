"""Colorbar sub-figure component"""
import bokeh.plotting
import bokeh.models
from forest.colors import one_way_connect, ColorSpec


class Colorbars:
    """Helper to layout/maintain colorbars"""
    def __init__(self):
        # Dimensions
        padding = 5
        margin = 20
        colorbar_height = 20
        plot_height = colorbar_height + 30
        plot_width = 500

        # LinearColorMapper
        self.color_mapper = bokeh.models.LinearColorMapper(
                low=0,
                high=1,
                palette=bokeh.palettes.Plasma[256])

        # Colorbar
        self.colorbar = bokeh.models.ColorBar(
            color_mapper=self.color_mapper,
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

    def connect(self, store):
        one_way_connect(self, store)
        return self

    def render(self, props):
        # Make ColorSpec from props
        # TODO: Tidy-up awkward mapping from props to ColorSpec
        kwargs = {k: v for k, v in props.items()
                  if k in ["name", "number", "low", "high", "reverse"]}
        if "invisible_min" in props:
            kwargs["low_visible"] = not props["invisible_min"]
        if "invisible_max" in props:
            kwargs["high_visible"] = not props["invisible_max"]
        spec = ColorSpec(**kwargs)
        print(f"forest.components.Colorbars {spec}")
        spec.apply(self.color_mapper)
