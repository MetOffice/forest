"""Capture feedback related to displayed data"""
import bokeh.layouts
import bokeh.models
from observe import Observable


SUBMIT = "SUBMIT"


class CSV(object):
    """Serialise survey results"""
    def insert_record(self, record):
        print("CSV: {}".format(record))


class Tool(Observable):
    def __init__(self):
        texts = [
            "Does the model agree with observations?",
            "If not, explain why."
        ]
        self.dropdown = bokeh.models.Dropdown(
                label="Yes/No",
                menu=[("Yes", "y"), ("No", "n")])
        def on_change(attr, old, new):
            for label, value in self.dropdown.menu:
                if value == new:
                    self.dropdown.label = label
        self.dropdown.on_change("value", on_change)
        self.text_area_input = bokeh.models.TextAreaInput(
                title="Type some text",
                rows=5)
        self.widgets = [
            bokeh.models.Div(text="<h1>Survey</h1>"),
            bokeh.models.Div(text=texts[0]),
            self.dropdown,
            bokeh.models.Div(text=texts[1]),
            self.text_area_input
        ]
        self.buttons = {
            "submit": bokeh.models.Button(
                label="Submit")
        }
        self.callbacks = {
            "submit": self.on_submit,
        }
        for key, on_click in self.callbacks.items():
            self.buttons[key].on_click(on_click)

        self.welcome = Welcome()
        self.welcome.subscribe(self.on_begin)
        self.confirm = Confirm()
        self.confirm.subscribe(self.on_save)
        self.layout = bokeh.layouts.column(
                *self.welcome.children)
        super().__init__()

    def on_save(self, action):
        print(action)

    def on_begin(self, action):
        self.layout.children = self.widgets + [
            self.buttons["submit"]
        ]

    def on_submit(self):
        answers = [
            self.dropdown.value,
            self.text_area_input.value
        ]
        self.layout.children = self.confirm.render(answers)


class Welcome(Observable):
    def __init__(self):
        self.div = bokeh.models.Div(
                text="""
        <h1>User feedback</h1>
        <p>Greetings! This survey aims to gather
        vital information on the quality of our
        forecasting systems. We are very pleased
        you've taken the time to fill it in.</p>
        """)
        self.button =  bokeh.models.Button(
                label="Begin")
        self.contact = bokeh.models.Div(
                text="""
        <p>If you have any queries related to the
        survey please <a href="mailto:andrew.hartley"metoffice.gov.uk>contact us</a></p>
        """)
        self.button.on_click(self.on_begin)
        self.children = [
            self.div,
            self.button,
            self.contact]
        super().__init__()

    def on_begin(self):
        self.notify({"kind": "BEGIN"})


class Confirm(Observable):
    def __init__(self):
        self.template = """
        <h1>Thank you</h1>
        <p>Your feedback is very important to us. We
        use the aggregated knowledge provided to
        improve our understanding of our systems
        beyond objective metrics.</p>
        <p>Your answers:</p>
        <ul>
            <li>Question 1: {}</li>
            <li>Question 2: {}</li>
        </ul>
        """
        self.div = bokeh.models.Div(text="")
        self.buttons = {
            "edit": bokeh.models.Button(label="Edit"),
            "save": bokeh.models.Button(label="Save")}
        self.buttons["edit"].on_click(self.on_edit)
        self.buttons["save"].on_click(self.on_save)
        self.contact = bokeh.models.Div(
                text="""
        <p>If you have any queries related to the
        survey please <a href="mailto:andrew.hartley"metoffice.gov.uk>contact us</a></p>
        """)
        super().__init__()

    def render(self, answers):
        self.div.text = self.template.format(*answers)
        return [
            self.div,
            self.buttons["edit"],
            self.buttons["save"],
            self.contact
        ]

    def on_edit(self):
        self.notify({"kind": "EDIT"})

    def on_save(self):
        self.notify({"kind": "SAVE"})
