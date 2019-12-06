"""Matplotlib-esque usage of barbs in Typescript/Bokeh

Additional comment
"""
import bokeh.plotting
from forest import wind # Magic to extend bokeh.Figure


def main():
    """Example using forest.wind.Barb"""
    x=[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
    y=[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
    u = [i*10 for i in x]
    v = [j*10 for j in y]
    figure = bokeh.plotting.figure()
    figure.barb(x=x, y=y, u=u, v=v)
    document = bokeh.plotting.curdoc()
    document.add_root(figure)


if __name__.startswith("bk"):
    main()
