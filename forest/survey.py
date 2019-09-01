"""Capture feedback related to displayed data"""
import bokeh.layouts
import bokeh.models
from observe import Observable


SUBMIT = "SUBMIT"
BEGIN = "BEGIN"
CONTACT_TEXT = """
<p>If you have any queries related to the survey please
<a href="mailto:andrew.hartley@metoffice.gov.uk">contact us</a>
</p>
"""


class CSV(object):
    """Serialise survey results"""
    def insert_record(self, record):
        print("CSV: {}".format(record))


class Tool(Observable):
    def __init__(self):
        self.welcome = Welcome()
        self.welcome.subscribe(self.on_action)
        self.questions = Questions()
        self.questions.subscribe(self.on_action)
        self.confirm = Confirm()
        self.confirm.subscribe(self.on_action)
        self.layout = bokeh.layouts.column(
                *self.welcome.children)
        super().__init__()

    def on_action(self, action):
        print(action)
        kind = action["kind"]
        if kind == BEGIN:
            self.layout.children = self.questions.children
        elif kind == SUBMIT:
            answers = action["payload"]["answers"]
            self.layout.children = self.confirm.render(answers)


class Questions(Observable):
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
        self.children = self.widgets + [
            self.buttons["submit"]
        ]
        super().__init__()

    def on_submit(self):
        answers = [
            self.dropdown.value,
            self.text_area_input.value
        ]
        self.notify({
            "kind": "SUBMIT",
            "payload": {"answers": answers}})


class Welcome(Observable):
    def __init__(self):
        self.div = bokeh.models.Div(
                text="""
        <h1>Greetings!</h1>
        <p>This survey aims to gather
        vital information on the quality of our
        forecasting systems. We are very pleased
        you've taken the time to fill it in.</p>
        """)
        self.buttons = {
            "begin": bokeh.models.Button(
                label="Begin"),
            "results": bokeh.models.Button(
                label="View results")}
        self.contact = bokeh.models.Div(
                text=CONTACT_TEXT)
        self.buttons["begin"].on_click(self.on_begin)
        self.buttons["results"].on_click(self.on_results)
        self.children = [
            self.div,
            self.buttons["begin"],
            self.buttons["results"],
            self.contact]
        super().__init__()

    def on_begin(self):
        self.notify({"kind": BEGIN})

    def on_results(self):
        self.notify({"kind": "RESULTS"})


class Confirm(Observable):
    def __init__(self):
        self.template = """
        <h1>Thank you</h1>
        <p>Your feedback is very important to us. We
        use the information provided to improve our
        understanding of our systems beyond that
        which can be derived by objective metrics.</p>
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
                text=CONTACT_TEXT)
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
