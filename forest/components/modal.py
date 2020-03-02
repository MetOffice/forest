import bokeh.models
import bokeh.layouts
from forest.observe import Observable
from forest import layers


class Modal(Observable):
    """Modal component"""
    def __init__(self):
        div = bokeh.models.Div(text="Add layer",
                               css_classes=["custom"],
                               sizing_mode="stretch_width")
        self.select = bokeh.models.Select(options=[])
        buttons = (
            bokeh.models.Button(label="Save"),
            bokeh.models.Button(label="Exit"))
        buttons[0].on_click(self.on_save)
        for button in buttons:
            custom_js = bokeh.models.CustomJS(code="""
                let el = document.getElementById("modal");
                el.style.visibility = "hidden";
            """)
            button.js_on_click(custom_js)
        self.layout = bokeh.layouts.column(
            div,
            self.select,
            bokeh.layouts.row(*buttons, sizing_mode="stretch_width"),
            name="modal",
            sizing_mode="stretch_width")
        super().__init__()

    def connect(self, store):
        store.add_subscriber(self.render)
        self.add_subscriber(store.dispatch)
        return self

    def render(self, state):
        self.select.options = self.to_props(state)
        if len(self.select.options) > 0:
            if self.select.value == "":
                self.select.value = self.select.options[0]

    def to_props(self, state):
        return [name for name, _ in state.get("patterns", [])]

    def on_save(self):
        name = self.select.value
        action = layers.add_layer(name)
        self.notify(action)
