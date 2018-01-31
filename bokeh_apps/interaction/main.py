from random import random

from bokeh.layouts import column, row
from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc

# create a plot and style its properties
p = figure(x_range=(0, 100), y_range=(0, 100), toolbar_location=None)
p.border_fill_color = 'black'
p.background_fill_color = 'black'
p.outline_line_color = None
p.grid.grid_line_color = None

# add a text renderer to our plot (no data yet)
r = p.text(x=[], y=[], text=[], text_color=[], text_font_size="20pt",
           text_baseline="middle", text_align="center")

i = 0

ds = r.data_source

# create a callback that will add a number in a random location
def one():
    update([getOne()])

def one_hundred():
    update([getOne() for i in range(100)])


def getOne():
    global i
    i += 1
    return {
        'x':random()*70 + 15,
        'y':random()*70 + 15,
        'text_color': RdYlBu3[i%3],
        'text':str(i)
    }

def update(to_add):
    new_data = dict()
    for key in ['x','y','text_color','text']:
        new_data[key] = ds.data[key]
        for item in to_add:
            new_data[key] = new_data[key] + [item[key]]
    ds.data = new_data

# add a button widget and configure with the call back
one_more_button = Button(label="Press Me")
one_more_button.on_click(one)

one_hundred_more_button = Button(label="Give me 100")
one_hundred_more_button.on_click(one_hundred)

# put the button and plot in a layout and add to the document
curdoc().add_root(
    column(
        row(one_more_button, one_hundred_more_button),
        p
    )
)