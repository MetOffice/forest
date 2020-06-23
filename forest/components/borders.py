"""User-defined border overlays"""
import bokeh.models


class Component:
    def __init__(self):
        # Lakes
        for figure in figures:
            add_feature(figure, data.LAKES, color="lightblue")

        features = []
        for figure in figures:
            features += [
                add_feature(figure, data.COASTLINES),
                add_feature(figure, data.BORDERS)]

        # Disputed borders
        for figure in figures:
            add_feature(figure, data.DISPUTED, color="red")

        toggle = bokeh.models.CheckboxGroup(
                labels=["Coastlines"],
                active=[0],
                width=135)

        def on_change(attr, old, new):
            if len(new) == 1:
                for feature in features:
                    feature.visible = True
            else:
                for feature in features:
                    feature.visible = False

        toggle.on_change("active", on_change)

        dropdown = bokeh.models.Dropdown(
                label="Color",
                menu=[
                    ("Black", "black"),
                    ("White", "white")],
                width=50)
        autolabel(dropdown)

        def on_change(event):
            for feature in features:
                feature.glyph.line_color = new

        dropdown.on_click(on_change)

        div = bokeh.models.Div(text="", width=10)
        border_row = bokeh.layouts.row(
            bokeh.layouts.column(toggle),
            bokeh.layouts.column(div),
            bokeh.layouts.column(dropdown))


def add_feature(figure, data, color="black"):
    source = bokeh.models.ColumnDataSource(data)
    return figure.multi_line(
        xs="xs",
        ys="ys",
        source=source,
        color=color)
