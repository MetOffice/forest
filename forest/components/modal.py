import bokeh.models
import bokeh.layouts
from forest.observe import Observable
from forest import layers


class Modal(Observable):
    """Modal component"""
    def __init__(self):
        self.div = bokeh.models.Div(text="Add layer",
                               css_classes=["custom"],
                               sizing_mode="stretch_width")
        self.inputs = {}

        # Text-input to name layer
        self.inputs["name"] = bokeh.models.TextInput(
            title="Name:",
            placeholder="Enter text")

        # Select layer settings, e.g. dataset, variable, etc.
        self.selects = {}
        self.selects["dataset"] = bokeh.models.Select(
            title="Dataset:",
            options=[])
        self.selects["variable"] = bokeh.models.Select(
            title="Variable:",
            options=[])

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
            self.div,
            self.inputs["name"],
            self.selects["dataset"],
            self.selects["variable"],
            bokeh.layouts.row(*buttons, sizing_mode="stretch_width"),
            name="modal",
            sizing_mode="stretch_width")
        super().__init__()

    def connect(self, store):
        store.add_subscriber(self.render)
        self.add_subscriber(store.dispatch)
        return self

    def render(self, state):
        # Configure title
        mode = state.get("layers", {}).get("mode", {}).get("state", "add")
        self.div.text = {"edit": "Edit layer"}.get(mode, "Add layer")

        # Set name for layer, e.g. layer-0
        node = state
        for key in ("layers", "mode"):
            node = node.get(key, {})
        mode = node.get("state", "add")
        if mode == "edit":
            index = node["index"]
            label = state["layers"]["index"][index]["label"]
            self.inputs["name"].value = label
        else:
            self.inputs["name"].value = "layer-0"

        # Configure available datasets
        self.selects["dataset"].options = self.to_props(state)
        if len(self.selects["dataset"].options) > 0:
            if self.selects["dataset"].value == "":
                self.selects["dataset"].value = self.selects["dataset"].options[0]

        self.selects["variable"].options = state.get("variables", [])

    def to_props(self, state):
        return [name for name, _ in state.get("patterns", [])]

    def on_save(self):
        # Send settings to forest.layers to process
        settings = {
            "label": self.inputs["name"].value,
            "dataset": self.selects["dataset"].value,
            "variable": self.selects["variable"].value
        }
        self.notify(layers.on_save(settings))
