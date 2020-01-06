"""
Presets
-------

User configured settings can be saved, edited and deleted.

UI components
~~~~~~~~~~~~~

.. autoclass:: PresetUI
   :members:

Reducer
~~~~~~~

.. autofunction:: reducer

Middleware
~~~~~~~~~~

Middleware pre-processes actions prior to the reducer

.. autofunction:: middleware

Helpers
~~~~~~~

.. autofunction:: state_to_props

Actions
~~~~~~~

A simple grammar used to communicate between components.

.. autofunction:: save_preset

.. autofunction:: load_preset

.. autofunction:: remove_preset

.. autofunction:: set_default_mode

.. autofunction:: set_edit_mode

.. autofunction:: set_edit_label

.. autofunction:: on_save

.. autofunction:: on_edit

.. autofunction:: on_new

.. autofunction:: on_cancel

"""
import copy
import bokeh.models
import bokeh.layouts
from forest.observe import Observable
from forest import redux, rx

# Action kinds
PRESET_SAVE = "PRESET_SAVE"
PRESET_LOAD = "PRESET_LOAD"
PRESET_REMOVE = "PRESET_REMOVE"
PRESET_SET_META = "PRESET_SET_META"
PRESET_ON_SAVE = "PRESET_ON_SAVE"
PRESET_ON_LOAD = "PRESET_ON_LOAD"
PRESET_ON_NEW = "PRESET_ON_NEW"
PRESET_ON_EDIT = "PRESET_ON_EDIT"
PRESET_ON_CANCEL = "PRESET_ON_CANCEL"

# Display modes
DEFAULT = "DEFAULT"
EDIT = "EDIT"


def save_preset(label):
    """Action to save a preset"""
    return {"kind": PRESET_SAVE, "payload": label}


def load_preset(label):
    """Action to load a preset by label"""
    return {"kind": PRESET_LOAD, "payload": label}


def remove_preset():
    """Action to remove a preset"""
    return {"kind": PRESET_REMOVE}


def set_default_mode():
    """Action to select default display mode"""
    return {"kind": PRESET_SET_META, "meta": {"mode": DEFAULT}}


def set_edit_mode():
    """Action to select edit display mode"""
    return {"kind": PRESET_SET_META, "meta": {"mode": EDIT}}


def set_edit_label(label):
    """Action to set edit mode label"""
    return {"kind": PRESET_SET_META, "meta": {"label": label}}


def on_save(label):
    """Action to signal save clicked"""
    return {"kind": PRESET_ON_SAVE, "payload": label}


def on_load(label):
    """Action to signal load clicked"""
    return {"kind": PRESET_ON_LOAD, "payload": label}


def on_edit():
    """Action to signal edit clicked"""
    return {"kind": PRESET_ON_EDIT}


def on_new():
    """Action to signal new clicked"""
    return {"kind": PRESET_ON_NEW}


def on_cancel():
    """Action to signal cancel clicked"""
    return {"kind": PRESET_ON_CANCEL}


def state_to_props(state):
    """Converts application state to props used by user interface"""
    query = Query(state)
    return query.labels, query.display_mode, query.edit_label


class Middleware:
    def __init__(self, storage):
        self.storage = storage

    def __call__(self, store, action):
        kind = action["kind"]
        if kind == PRESET_ON_SAVE:
            label = action["payload"]
            settings = copy.deepcopy(store.state.get("colorbar", {}))
            self.storage.save(label, settings)
        yield action


class Storage:
    """Store preset settings in-memory"""
    def __init__(self):
        self._records = {}

    def labels(self):
        return [key for key in self._records.keys()]

    def save(self, label, settings):
        self._records[label] = settings

    def load(self, label):
        return self._records[label]


def middleware(store, action):
    """Presets middleware

    Generates actions given current state and an incoming action. Encapsulates
    the business logic surrounding saving, editing and creating presets.

    """
    kind = action["kind"]
    if kind == PRESET_ON_SAVE:
        yield save_preset(action["payload"])
        yield set_default_mode()
    elif kind == PRESET_ON_LOAD:
        # Translate on_load() to load_preset() action
        yield load_preset(action["payload"])
    elif kind == PRESET_ON_CANCEL:
        yield set_default_mode()
    elif kind == PRESET_ON_EDIT:
        yield set_edit_label(Query(store.state).label)
        yield set_edit_mode()
    elif kind == PRESET_ON_NEW:
        yield set_edit_label("")
        yield set_edit_mode()
    else:
        yield action


def reducer(state, action):
    """Presets reducer

    :returns: next state
    """
    state = copy.deepcopy(state)
    kind = action["kind"]
    if kind == PRESET_SAVE:
        label = action["payload"]
        try:
            uid = Query(state).find_id(label)
        except IDNotFound:
            uid = new_id(Query(state).all_ids)
        if "presets" not in state:
            state["presets"] = {}
        if "labels" not in state["presets"]:
            state["presets"]["labels"] = {}
        if "settings" not in state["presets"]:
            state["presets"]["settings"] = {}
        state["presets"]["labels"][uid] = label
        if "colorbar" in state:
            settings = copy.deepcopy(state["colorbar"])
        else:
            settings = {}
        state["presets"]["settings"][uid] = settings

    elif kind == PRESET_LOAD:
        label = action["payload"]
        uid = Query(state).find_id(label)
        settings = copy.deepcopy(state["presets"]["settings"][uid])
        print(label, uid, settings)
        state["colorbar"] = settings
        state["presets"]["active"] = uid
    elif kind == PRESET_REMOVE:
        uid = state["presets"]["active"]
        del state["presets"]["labels"][uid]
        del state["presets"]["active"]
    elif kind == PRESET_SET_META:
        if "presets" not in state:
            state["presets"] = {}
        if "meta" not in state["presets"]:
            state["presets"]["meta"] = {}
        state["presets"]["meta"].update(action["meta"])
    return state


class IDNotFound(Exception):
    pass


class Query:
    """Helper to retrieve values stored in state"""
    def __init__(self, state):
        self.state = state

    @property
    def labels(self):
        return list(self.state.get("presets", {}).get("labels", {}).values())

    @property
    def display_mode(self):
        return self.state.get("presets", {}).get("meta", {}).get("mode", DEFAULT)

    @property
    def edit_label(self):
        """Label used by UI to allow user to save/edit"""
        return self.state.get("presets", {}).get("meta", {}).get("label", "")

    @property
    def all_ids(self):
        return set(self.state.get("presets", {}).get("labels", {}).keys())

    def find_id(self, label):
        labels = self.state.get("presets", {}).get("labels", {})
        for id, _label in labels.items():
            if _label == label:
                return id
        raise IDNotFound("'{}' not found".format(label))

    @property
    def label(self):
        if "presets" not in self.state:
            return ""
        if "active" not in self.state["presets"]:
            return ""
        uid = self.state["presets"]["active"]
        return self.state["presets"]["labels"][uid]


def new_id(ids):
    if len(ids) == 0:
        return 0
    return max(ids) + 1


class PresetUI(Observable):
    """User interface to load/save/edit presets

    >>> preset_ui = PresetUI().connect(store)

    """
    def __init__(self):
        self.select = bokeh.models.Select()
        self.select.on_change("value", self.on_load)
        self.text_input = bokeh.models.TextInput(placeholder="Save name")
        self.buttons = {
            "edit": bokeh.models.Button(label="Edit"),
            "new": bokeh.models.Button(label="New"),
            "cancel": bokeh.models.Button(label="Cancel"),
            "save": bokeh.models.Button(label="Save"),
        }
        self.buttons["save"].on_click(self.on_save)
        self.buttons["new"].on_click(self.on_new)
        self.buttons["edit"].on_click(self.on_edit)
        self.buttons["cancel"].on_click(self.on_cancel)
        width = 320
        self.children = {
            DEFAULT: [
                self.select, self.buttons["edit"], self.buttons["new"]
            ],
            EDIT: [
                self.text_input, self.buttons["cancel"], self.buttons["save"]
            ]
        }
        self.rows = {
                "title": bokeh.layouts.row(
                    bokeh.models.Div(text="Presets:"),
                    width=width),
                "content": bokeh.layouts.row(
                    self.children[DEFAULT],
                    width=width)}
        self.layout = bokeh.layouts.column(
                self.rows["title"],
                self.rows["content"])
        super().__init__()

    def connect(self, store):
        """Convenient method to map state to props needed by render"""
        self.subscribe(store.dispatch)
        stream = (rx.Stream()
                    .listen_to(store)
                    .map(state_to_props)
                    .filter(lambda x: x is not None)
                    .distinct())
        stream.map(lambda props: self.render(*props))
        return self

    def on_save(self):
        """Notify listeners that a save action has taken place"""
        label = self.text_input.value
        if label != "":
            self.notify(on_save(label))

    def on_load(self, attr, old, new):
        """Notify listeners that a load action has taken place"""
        self.notify(on_load(new))

    def on_new(self):
        self.notify(on_new())

    def on_edit(self):
        self.notify(on_edit())

    def on_cancel(self):
        self.notify(on_cancel())

    def render(self, labels, mode, edit_label):
        # TODO: Add support for DEFAULT/EDIT mode layouts
        self.rows["content"].children = self.children[mode]
        self.select.options = list(sorted(labels))
        self.text_input.value = edit_label
