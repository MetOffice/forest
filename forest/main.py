#!/usr/bin/env python
import argparse
import bokeh.plotting
import geo
import sys
import forest


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    return parser.parse_args(args=args)


def main():
    args = parse_args(sys.argv)
    loaders = forest.loaders(args.files)

    # Full screen map
    lon_range = (0, 30)
    lat_range = (0, 30)
    x_range, y_range = geo.web_mercator(
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
    figure.axis.visible = False
    figure.toolbar.logo = None
    figure.toolbar_location = None
    figure.min_border = 0
    figure.add_tile(tile)
    document = bokeh.plotting.curdoc()
    root = bokeh.layouts.row(
            figure,
            sizing_mode="stretch_both")
    document.add_root(root)


if __name__.startswith('bk'):
    main()
