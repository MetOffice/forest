"""Utility functions, decorators and classes"""


__all__ = [
    "autolabel"
]


def autolabel(dropdown):
    """Automatically set Dropdown label on_click"""
    def callback(event):
        for label, _value in dropdown.menu:
            if event.item == _value:
                dropdown.label = label
    dropdown.on_click(callback)
    return callback
