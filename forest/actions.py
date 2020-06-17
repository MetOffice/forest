"""Collection of actions"""
from dataclasses import dataclass, field, asdict
from forest.colors import (
    set_palette_name,
    set_user_high
)
from forest.layers import (
    save_layer
)


@dataclass
class Action:
    kind: str
    payload: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


HTML_LOADED = "BOKEH_HTML_LOADED"
NO_ACTION = "NO_ACTION"


def no_action():
    return {"kind": NO_ACTION}


def html_loaded():
    return Action(HTML_LOADED)
