"""
Presets
-------

User configured settings can be saved, edited and deleted.


.. autoclass:: PresetUI
   :members:

.. autofunction:: reducer

.. autofunction:: save_preset

.. autofunction:: load_preset

"""
import copy
import bokeh.models
import bokeh.layouts
from forest.observe import Observable
from forest import redux, rx

# Action kinds
SAVE_PRESET = "SAVE_PRESET"
LOAD_PRESET = "LOAD_PRESET"
RENAME_PRESET = "RENAME_PRESET"
REMOVE_PRESET = "REMOVE_PRESET"
PRESET_SET_META = "PRESET_SET_META"
PRESET_ON_SAVE = "PRESET_ON_SAVE"
PRESET_ON_CANCEL = "PRESET_ON_CANCEL"

# Display modes
DEFAULT = "DEFAULT"
EDIT = "EDIT"


def save_preset(label):
    """Action to save a preset"""
    return {"kind": SAVE_PRESET, "payload": label}


def load_preset(label):
    """Action to load a preset by label"""
    return {"kind": LOAD_PRESET, "payload": label}


def rename_preset(label):
    """Action to rename a preset"""
    return {"kind": RENAME_PRESET, "payload": label}


def remove_preset():
    """Action to remove a preset"""
    return {"kind": REMOVE_PRESET}


def set_default_mode():
    return {"kind": PRESET_SET_META, "meta": {"mode": DEFAULT}}


def set_edit_mode():
    return {"kind": PRESET_SET_META, "meta": {"mode": EDIT}}


def on_save(label):
    return {"kind": PRESET_ON_SAVE, "payload": label}


def on_cancel():
    return {"kind": PRESET_ON_CANCEL}


def state_to_props(state):
    options = list(state.get("presets", {}).get("labels", {}).values())
    mode = state.get("presets", {}).get("meta", {}).get("mode", DEFAULT)
    return options, mode


@redux.middleware
def middleware(store, next_dispatch, action):
    kind = action["kind"]
    if kind == PRESET_ON_SAVE:
        next_dispatch(save_preset(action["payload"]))
        next_dispatch(set_default_mode())
    elif kind == PRESET_ON_CANCEL:
        next_dispatch(set_default_mode())
    else:
        next_dispatch(action)


def reducer(state, action):
    state = copy.deepcopy(state)
    kind = action["kind"]
    if kind == SAVE_PRESET:
        label = action["payload"]
        try:
            uid = find_id(state, label)
        except IDNotFound:
            uid = new_id(all_ids(state))
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

    elif kind == LOAD_PRESET:
        label = action["payload"]
        uid = find_id(state, label)
        settings = copy.deepcopy(state["presets"]["settings"][uid])
        print(label, uid, settings)
        state["colorbar"] = settings
        state["presets"]["active"] = uid
    elif kind == RENAME_PRESET:
        uid = state["presets"]["active"]
        state["presets"]["labels"][uid] = action["payload"]
    elif kind == REMOVE_PRESET:
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


def find_id(state, label):
    labels = state.get("presets", {}).get("labels", {})
    for id, _label in labels.items():
        if _label == label:
            return id
    raise IDNotFound("'{}' not found".format(label))


def all_ids(state):
    return set(state.get("presets", {}).get("labels", {}).keys())


def new_id(ids):
    if len(ids) == 0:
        return 0
    return max(ids) + 1


class PresetUI(Observable):
    """User interface to load/save/edit presets"""
    def __init__(self):
        self.select = bokeh.models.Select()
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

    def on_load(self):
        """Notify listeners that a load action has taken place"""
        label = self.select.value
        self.notify(load_preset(label))

    def on_new(self):
        self.notify(set_edit_mode())

    def on_edit(self):
        self.notify(set_edit_mode())

    def on_cancel(self):
        self.notify(on_cancel())

    def render(self, labels, mode):
        # TODO: Add support for DEFAULT/EDIT mode layouts
        self.rows["content"].children = self.children[mode]
        self.select.options = list(sorted(labels))
