"""Capture feedback related to displayed data"""
import bokeh.layouts
import bokeh.models
from observe import Observable


SUBMIT = "SUBMIT"


class CSV(object):
    """Serialise survey results"""
    def insert_record(self, record):
        print(record)


class Tool(Observable):
    def __init__(self):
        self.div = bokeh.models.Div(text="Survey")
        self.buttons = {
            "submit": bokeh.models.Button()
        }
        self.buttons["submit"].on_click(self.on_save)
        self.layout = bokeh.layouts.column(
            self.div,
            self.buttons["submit"])
        super().__init__()

    def on_save(self):
        self.notify(submit())


def submit():
    return {
        "kind": SUBMIT,
        "payload": {}
    }
