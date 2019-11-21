from forest import colors
import bokeh.models


def test_color_controls():
    color_mapper = bokeh.models.LinearColorMapper()
    name = "Accent"
    number = 3
    controls = colors.Controls(color_mapper, name, number)
    controls.render()
