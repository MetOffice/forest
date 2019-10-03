"""Key press interactivity"""
from forest.observe import Observable
from forest.export import export


__all__ = []


KEY_PRESS = "KEY_PRESS"


def press(code):
    """Key press action creator"""
    return {
        "kind": KEY_PRESS,
        "payload": {
            "code": code
        }
    }


@export
class KeyPress(Observable):
    """Key press observable"""
    def on_change(self, attr, old, new):
        code = new[0]
        self.notify(press(code))
