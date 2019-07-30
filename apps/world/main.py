import bokeh.plotting
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import forest.geo


def main():
    lon_range = (0, 30)
    lat_range = (0, 30)
    x_range, y_range = forest.geo.web_mercator(
        lon_range,
        lat_range)
    figure = bokeh.plotting.figure(
        x_range=x_range,
        y_range=y_range,
        x_axis_type="mercator",
        y_axis_type="mercator",
        active_scroll="wheel_zoom")
    tile = bokeh.models.WMTSTileSource(
        url="https://maps.wikimedia.org/osm-intl/{Z}/{X}/{Y}.png",
        attribution=""
    )
    figure.add_tile(tile)
    figure.axis.visible = False
    figure.toolbar.logo = None
    figure.toolbar_location = None
    figure.min_border = 0
    row = bokeh.layouts.row(figure, sizing_mode="stretch_both")
    document = bokeh.plotting.curdoc()
    document.title = "FOREST"
    document.add_root(row)


if __name__.startswith("bk"):
    main()
