import bokeh.models
import bokeh.layouts


class Modal:
    def __init__(self):
        div = bokeh.models.Div(text="Add layer",
                               css_classes=["custom"],
                               sizing_mode="stretch_width")
        self.select = bokeh.models.Select(options=[])
        buttons = (
            bokeh.models.Button(label="Save"),
            bokeh.models.Button(label="Exit"))
        for button in buttons:
            custom_js = bokeh.models.CustomJS(code="""
                let el = document.getElementById("dialogue");
                el.style.visibility = "hidden";
                console.log(el);
            """)
            button.js_on_click(custom_js)
        self.layout = bokeh.layouts.column(
            div,
            self.select,
            bokeh.layouts.row(*buttons, sizing_mode="stretch_width"),
            name="modal",
            sizing_mode="stretch_width")

    def connect(self, store):
        store.add_subscriber(self.render)
        return self

    def render(self, state):
        options = self.to_props(state)
        print(options)
        self.select.options = options

    def to_props(self, state):
        labels = state.get("layers", {}).get("labels", [])
        return [label for label in labels if label is not None]
