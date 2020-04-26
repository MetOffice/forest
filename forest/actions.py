"""Collection of actions"""
from forest.colors import (
    set_palette_name,
    set_user_high
)
from forest.layers import (
    save_layer
)


NO_ACTION = "NO_ACTION"


def no_action():
    return {"kind": NO_ACTION}
