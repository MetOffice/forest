
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
import numpy

def figure_to_object(fig ,linkto=None):
    
    canvas =  FigureCanvas(fig)
    width, height = fig.get_size_inches() * fig.get_dpi()
    canvas.draw()
    
    # Get the RGB buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = numpy.fromstring ( fig.canvas.tostring_rgb(), dtype=numpy.uint8 )
    buf.shape = ( w, h,3 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll ( buf, 3, axis = 2 )
    
    img = buf
    if img.ndim > 2: # could also be img.dtype == np.uint8
        if img.shape[2] == 3: # alpha channel not included
            img = numpy.dstack([img, numpy.ones(img.shape[:2], numpy.uint8) * 255])
        img = numpy.squeeze(img.view(numpy.uint32))
    
    img = numpy.flip(img,0 )

    
    x_range = linkto.x_range if linkto else (0,10)
    y_range = linkto.y_range if linkto else (0,10)
    p = figure(x_range=x_range, y_range=y_range)

    # must give a vector of images
    p.image_rgba(image=[img], x=0, y=0, dw=10, dh=10)
    p.axis.visible = False
    
    return p
