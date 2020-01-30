"""Utility functions, decorators and classes"""


__all__ = [
    "autolabel"
]


def autolabel(dropdown):
    """Automatically set Dropdown label on_click"""
    def callback(attr, old, new):
        for label, _value in dropdown.menu:
            if new == _value:
                dropdown.label = label
    dropdown.on_change("value", callback)
    return callback
