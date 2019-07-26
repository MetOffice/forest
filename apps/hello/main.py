import bokeh.models
import bokeh.layouts
import bokeh.plotting


def main():
    button = bokeh.models.Button()
    div = bokeh.models.Div()
    def on_click():
        div.text = "Hello, World!"
    button.on_click(on_click)
    root = bokeh.layouts.column(button, div)
    document = bokeh.plotting.curdoc()
    document.add_root(root)


if __name__.startswith("bk"):
    main()
