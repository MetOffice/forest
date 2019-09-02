"""Capture feedback related to displayed data"""
import bokeh.layouts
import bokeh.models
import copy
import json
import datetime as dt
from observe import Observable
from functools import wraps


EDIT = "EDIT"
BEGIN = "BEGIN"
CONFIRM = "CONFIRM"
QUESTION = "QUESTION"
SUBMIT = "SUBMIT"
SAVE = "SAVE"
SHOW_RESULTS = "SHOW_RESULTS"
SHOW_WELCOME = "SHOW_WELCOME"
RESULTS = "RESULTS"
WELCOME = "WELCOME"
CONTACT_TEXT = """
<p>If you have any queries related to the survey please
<a href="mailto:andrew.hartley@metoffice.gov.uk">contact us</a>
</p>
"""


class Store(Observable):
    def __init__(self, reducer, initial_state=None, middlewares=None):
        self.reducer = reducer
        self.state = initial_state if initial_state is not None else {}
        if middlewares is not None:
            mws = [m(self) for m in middlewares]
            f = self.dispatch
            for mw in reversed(mws):
                f = mw(f)
            self.dispatch = f
        super().__init__()

    def dispatch(self, action):
        self.state = self.reducer(self.state, action)
        self.notify(self.state)


def reducer(state, action):
    kind = action["kind"]
    state = copy.copy(state)
    if (kind == BEGIN) or (kind == EDIT):
        state["page"] = QUESTION
    elif kind == SUBMIT:
        state["page"] = CONFIRM
        state["answers"] = action["payload"]["answers"]
    elif (kind == SAVE) or (kind == SHOW_WELCOME):
        state["page"] = WELCOME
    elif kind == SHOW_RESULTS:
        state["page"] = RESULTS
    return state


def middleware(f):
    """Decorator to curry middleware call signature

    ..note:: Can decorate either a function or a method
    on a class
    """
    @wraps(f)
    def outer(*args):
        def inner(next_method):
            def inner_most(action):
                f(*args, next_method, action)
            return inner_most
        return inner
    return outer


def show_results():
    return {"kind": SHOW_RESULTS}


def show_welcome():
    return {"kind": SHOW_WELCOME}


class Database(object):
    """Serialise survey results"""
    def __init__(self, path):
        self.path = path
        self.records = []

    def insert(self, record):
        self.records.append(record)

    def flush(self):
        with open(self.path, "w") as stream:
            json.dump({"records": self.records}, stream)


class DatabaseMiddleware(object):
    """Intercept action(s) related to database operations"""
    def __init__(self, database):
        self.database = database

    @middleware
    def __call__(self, store, next_method, action):
        print(store, action, next_method)
        kind = action["kind"]
        if kind == SAVE:
            next_method(show_results())
        else:
            next_method(action)


class Tool(Observable):
    def __init__(self, database=None):
        self.database = database
        middlewares = []
        if database is not None:
            middlewares.append(DatabaseMiddleware(database))
        self.store = Store(
                reducer,
                initial_state={"page": WELCOME},
                middlewares=middlewares)
        self.store.subscribe(print)
        self.store.subscribe(self.render)
        self.welcome = Welcome()
        self.welcome.subscribe(self.store.dispatch)
        self.questions = Questions()
        self.questions.subscribe(self.store.dispatch)
        self.results = Results()
        self.results.subscribe(self.store.dispatch)
        self.confirm = Confirm()
        self.confirm.subscribe(self.store.dispatch)
        self.layout = bokeh.layouts.column(
                *self.welcome.children)
        super().__init__()

    def render(self, state):
        if "page" not in state:
            return
        page = state["page"]
        if page == WELCOME:
            self.layout.children = self.welcome.children
        elif page == QUESTION:
            self.layout.children = self.questions.children
        elif page == CONFIRM:
            self.layout.children = self.confirm.render(state["answers"])
        elif page == RESULTS:
            self.layout.children = self.results.children


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
                placeholder="Type some text",
                rows=5)
        self.widgets = [
            bokeh.models.Div(text="<h1>Survey</h1>"),
            bokeh.models.Div(text=texts[0]),
            self.dropdown,
            bokeh.models.Div(text=texts[1]),
            self.text_area_input,
            bokeh.models.Spinner(low=0, high=10, step=1, value=5),
        ]
        self.buttons = {
            "submit": bokeh.models.Button(
                label="Submit"),
            "welcome": bokeh.models.Button(
                label="Home")
        }
        self.callbacks = {
            "submit": self.on_submit,
            "welcome": self.on_welcome
        }
        for key, on_click in self.callbacks.items():
            self.buttons[key].on_click(on_click)
        self.children = self.widgets + [
            self.buttons["submit"],
            self.buttons["welcome"]
        ]
        super().__init__()

    def on_submit(self):
        answers = [
            self.dropdown.value,
            self.text_area_input.value
        ]
        self.notify({
            "kind": SUBMIT,
            "payload": {"answers": answers}})

    def on_welcome(self):
        self.notify(show_welcome())


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
        self.notify(show_results())


class Results(Observable):
    def __init__(self):
        self.button = bokeh.models.Button(label="Home")
        self.button.on_click(self.on_welcome)
        self.children = [
            bokeh.models.Div(text="<h1>Survey results</h1>"),
            bokeh.models.Div(text="Placeholder text"),
            self.button
        ]
        super().__init__()

    def on_welcome(self):
        self.notify(show_welcome())


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
        <p>If you would like to edit your answers
        please do so now, otherwise feel free to save
        your results.</p>
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
        self.notify({"kind": EDIT})

    def on_save(self):
        self.notify({"kind": SAVE})
