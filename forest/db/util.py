"""Utility functions, decorators and classes"""


__all__ = [
    "autolabel",
    "autowarn"
]


def autolabel(dropdown):
    """Automatically set Dropdown label on_click"""
    def callback(attr, old, new):
        for label, _value in dropdown.menu:
            if new == _value:
                dropdown.label = label
    dropdown.on_change("value", callback)
    return callback


def autowarn(dropdown):
    """Automatically change button type based on available menu choices"""
    def on_menu(attr, old, new):
        if dropdown.label in pluck_label(new):
            dropdown.button_type = "default"
        else:
            dropdown.button_type = "danger"

    def on_value(attr, old, new):
        label = find_label(dropdown.menu, new)
        if label in pluck_label(dropdown.menu):
            dropdown.button_type = "default"
        else:
            dropdown.button_type = "danger"

    dropdown.on_change("menu", on_menu)
    dropdown.on_change("value", on_value)
    return on_menu


def find_label(menu, value):
    for label, _value in menu:
        if value == _value:
            return label


def pluck_label(menu):
    return [l for l, _ in menu]
