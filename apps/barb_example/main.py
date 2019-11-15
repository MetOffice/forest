"""Matplotlib-esque usage of barbs in Typescript/Bokeh"""
import bokeh.plotting
from forest import wind # Magic to extend bokeh.Figure


def main():
    """Example using forest.wind.Barb"""
    figure = bokeh.plotting.figure()
    figure.barb(x=[0], y=[0]) #, u=[10], v=[10])
    figure.circle(x=[1, 2, 3], y=[1, 2, 3])
    document = bokeh.plotting.curdoc()
    document.add_root(figure)


if __name__.startswith("bk"):
    main()
