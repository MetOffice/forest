"""Capture feedback related to displayed data"""
import bokeh.layouts
import bokeh.models
import copy
import json
import datetime as dt
import jinja2
from observe import Observable
from functools import wraps


EDIT = "EDIT"
BEGIN = "BEGIN"
CONFIRM = "CONFIRM"
QUESTION = "QUESTION"
SUBMIT = "SUBMIT"
SAVE = "SAVE"
SET_RESULTS = "SET_RESULTS"
SHOW_RESULTS = "SHOW_RESULTS"
REFRESH_RESULTS = "REFRESH_RESULTS"
SHOW_WELCOME = "SHOW_WELCOME"
RESET_ANSWERS = "RESET_ANSWERS"
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
    elif kind == SHOW_WELCOME:
        state["page"] = WELCOME
    elif kind == SHOW_RESULTS:
        state["page"] = RESULTS
    elif kind == RESET_ANSWERS:
        state["answers"] = []
    elif kind == SET_RESULTS:
        state["results"] = action["payload"]["results"]
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


def reset_answers():
    return {"kind": RESET_ANSWERS}


def set_results(results):
    return {
            "kind": SET_RESULTS,
            "payload": {
                    "results": results
                }
            }


def refresh_results():
    return {"kind": REFRESH_RESULTS}


def show_results():
    return {"kind": SHOW_RESULTS}


def show_welcome():
    return {"kind": SHOW_WELCOME}


class Database(object):
    """Serialise survey results"""
    def __init__(self, path):
        self.path = path
        try:
            with open(self.path, "r") as stream:
                records = json.load(stream)["records"]
        except IOError:
            records = []
        self.records = records

    def insert(self, record):
        self.records.append(record)
        self.flush()

    def flush(self):
        with open(self.path, "w") as stream:
            json.dump({"records": self.records}, stream)


class DatabaseMiddleware(object):
    """Intercept action(s) related to database operations"""
    def __init__(self, database):
        self.database = database

    @middleware
    def __call__(self, store, next_method, action):
        kind = action["kind"]
        if kind == SAVE:
            self.database.insert({
                "timestamp": str(dt.datetime.now()),
                "answers": copy.copy(store.state["answers"])
            })
            next_method(set_results(self.database.records))
            next_method(show_results())
            next_method(reset_answers())
        elif kind == SHOW_RESULTS:
            next_method(set_results(self.database.records))
            next_method(action)
        elif kind == REFRESH_RESULTS:
            next_method(set_results(self.database.records))
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
            self.layout.children = self.questions.render(state)
        elif page == CONFIRM:
            self.layout.children = self.confirm.render(state["answers"])
        elif page == RESULTS:
            self.layout.children = self.results.render(state)


class Question(object):
    @classmethod
    def pose(cls, kind, *args, **kwargs):
        kind = kind.lower().replace(' ', '')
        for sub_cls in cls.__subclasses__():
            if sub_cls.__name__.lower() == kind:
                return sub_cls(*args, **kwargs)
        raise Exception("Unknown question: {}".format(kind))


class YesNo(Question):
    def __init__(self, text):
        self.label = "Yes/No"
        self.div = bokeh.models.Div(text=text)
        self.dropdown = bokeh.models.Dropdown(
                label=self.label,
                menu=[("Yes", "y"), ("No", "n")])
        def on_change(attr, old, new):
            for label, value in self.dropdown.menu:
                if value == new:
                    self.dropdown.label = label
        self.dropdown.on_change("value", on_change)
        self.children = [
            self.div,
            self.dropdown
        ]

    def reset(self):
        self.dropdown.label = self.label
        self.dropdown.value = None

    @property
    def answer(self):
        return self.dropdown.value

    @answer.setter
    def answer(self, value):
        self.dropdown.value = value


class LongAnswer(Question):
    def __init__(self, text):
        self.div = bokeh.models.Div(text=text)
        self.text_area_input = bokeh.models.TextAreaInput(
                placeholder="Type some text",
                rows=5)
        self.children = [
            self.div,
            self.text_area_input
        ]

    def reset(self):
        self.text_area_input.value = ""

    @property
    def answer(self):
        return self.text_area_input.value

    @answer.setter
    def answer(self, value):
        self.text_area_input.value = value


class ShortAnswer(Question):
    def __init__(self, text):
        self.div = bokeh.models.Div(text=text)
        self.text_input = bokeh.models.TextInput()
        self.children = [
            self.div,
            self.text_input
        ]

    def reset(self):
        self.text_input.value = ""

    @property
    def answer(self):
        return self.text_input.value

    @answer.setter
    def answer(self, value):
        self.text_input.value = value


class Questions(Observable):
    def __init__(self):
        self.questions = []
        for text, kind in ([
                ("What's your name?", "ShortAnswer"),
                ("Does the model agree with observations?", "YesNo"),
                ("If not, explain why.", "LongAnswer")]):
            self.questions.append(Question.pose(kind, text))
        self.widgets = [
            bokeh.models.Div(text="<h1>Survey</h1>"),
        ]
        for q in self.questions:
            self.widgets += q.children
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

    def render(self, state):
        if "answers" in state:
            answers = state["answers"]
            if len(answers) == 0:
                for question in self.questions:
                    question.reset()
            else:
                for question, answer in zip(
                        self.questions,
                        answers):
                    question.answer = answer
        return self.children

    def on_submit(self):
        answers = [q.answer for q in self.questions]
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
        width = 300
        height = 100

        # Question 1: Bar chart
        x_labels = ["Yes", "No"]
        self.figure = bokeh.plotting.figure(
                plot_width=width,
                plot_height=height,
                x_range=x_labels,
                toolbar_location=None,
                tools="",
                border_fill_alpha=0)
        self.bar_source = bokeh.models.ColumnDataSource({
            "x": x_labels,
            "top": [0, 0]
        })
        self.figure.vbar(
                x="x",
                top="top",
                width=0.9,
                source=self.bar_source)
        self.figure.y_range.start = 0

        # Question 2: Data table
        self.table_source = bokeh.models.ColumnDataSource(
                {"timestamp": [], "answer": []})
        columns = [
            bokeh.models.TableColumn(field="timestamp", title="Time"),
            bokeh.models.TableColumn(field="answer", title="Answer"),
        ]
        self.table = bokeh.models.DataTable(
                editable=False,
                source=self.table_source,
                columns=columns,
                width=width,
                height=height)

        self.buttons = {
            "home": bokeh.models.Button(
                label="Home",
                width=140),
            "refresh": bokeh.models.Button(
                label="Refresh",
                width=140),
        }
        self.buttons["home"].on_click(self.on_welcome)
        self.buttons["refresh"].on_click(self.on_refresh)
        self.children = [
            bokeh.models.Div(text="<h1>Survey results</h1>"),
            self.figure,
            self.table,
            bokeh.layouts.row(
                self.buttons["home"],
                self.buttons["refresh"])
        ]
        super().__init__()

    def render(self, state):
        # Bar chart
        yes, no = 0, 0
        for result in state.get("results", []):
            answers = result["answers"]
            if answers[1] == "y":
                yes += 1
            else:
                no += 1
        self.bar_source.data = {
            "x": ["Yes", "No"],
            "top": [yes, no]
        }

        # Data table
        times, texts = [], []
        for result in state.get("results", []):
            times.append(result["timestamp"])
            texts.append(result["answers"][2])
        self.table_source.data = {
            "timestamp": times,
            "answer": texts
        }
        return self.children

    def on_welcome(self):
        self.notify(show_welcome())

    def on_refresh(self):
        self.notify(refresh_results())


class Confirm(Observable):
    def __init__(self):
        self.template = jinja2.Template("""
        <h1>Thank you</h1>
        <p>Your feedback is very important to us. We
        use the information provided to improve our
        understanding of our systems beyond that
        which can be derived by objective metrics.</p>
        <p>Your answers:</p>
        <ul>
            {% for answer in answers %}
            <li>Q{{loop.index}}: {{answer}}</li>
            {% endfor %}
        </ul>
        <p>If you would like to edit your answers
        please do so now, otherwise feel free to save
        your results.</p>
        """)
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
        self.div.text = self.template.render(answers=answers)
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
