import bokeh.models


class Headline:
    """Summarise selected state"""
    def __init__(self):
        self._template = "<h3>{}</h3>"
        self.div = bokeh.models.Div(text=self._template.format(""),
                                       sizing_mode="stretch_width")
        self.layout = self.div

    def connect(self, store):
        store.add_subscriber(self.render)
        return self

    def render(self, state):
        labels = state.get("layers", {}).get("labels", [])
        content = ", ".join([label for label in labels if label is not None])
        self.div.text = self._template.format(content)
