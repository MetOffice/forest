'''SE Asia Sim. Im. and Himawari-8 Matplotlib app

 This script creates plots of simulated satellite imagery and
  Himawari-8 imagery for SE Asia using the Matplotlib plotting
  library to provide images to a Bokeh Server app.

'''
import os

import warnings
warnings.filterwarnings('ignore')

import imageio
import numpy as np
import bokeh.plotting
import forest.zoom


def main(bokeh_id):
    """Main program

    A stripped down version of Forest to see the woods from the trees

    The intention here is to write a prototype in an imperative style
    before extracting classes and methods with distinct responsibilities
    """
    print("making figure")
    figure = bokeh.plotting.figure(sizing_mode="stretch_both",
                                   match_aspect=True)

    # Plot a RGBA field from a single Himawari JPEG file
    jpeg = "~/s3/stephen-sea-public-london/himawari-8/LWIN11_201806122330.jpg"
    print("reading", jpeg)
    rgb = imageio.imread(jpeg)

    # Flip image to be right way up
    rgb = rgb[::-1]

    print("converting RGB to RGBA")
    rgba = forest.zoom.to_rgba(rgb)
    print("RGBA shape", rgba.shape)

    # Global extent coarse image
    dw = rgb.shape[1]
    dh = rgb.shape[0]
    coarse_image = forest.zoom.sub_sample(rgba, fraction=0.25)
    coarse_source = bokeh.models.ColumnDataSource({
        "image": [coarse_image],
        "x": [0],
        "y": [0],
        "dw": [dw],
        "dh": [dh]
    })
    figure.image_rgba(image="image",
                      x="x",
                      y="y",
                      dw="dw",
                      dh="dh",
                      source=coarse_source)

    rgba_zoom = forest.zoom.RGBAZoom(rgba)
    rgba_zoom.add_figure(figure)

    try:
        bokeh_mode = os.environ['BOKEH_MODE']
    except:
        bokeh_mode = 'server'
    print("bokeh_mode", bokeh_mode)
    if bokeh_mode == 'server':
        bokeh.plotting.curdoc().add_root(figure)
    elif bokeh_mode == 'cli':
        bokeh.io.show(figure)
    bokeh.plotting.curdoc().title = 'Model simulated imagery vs Himawari-8'



main(__name__)
