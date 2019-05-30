"""Utility functions, decorators and classes"""


def autolabel(dropdown):
    """Automatically set Dropdown label on_click"""
    def callback(value):
        for label, _value in dropdown.menu:
            if value == _value:
                dropdown.label = label
    dropdown.on_click(callback)
    return callback


def autowarn(dropdown):
    """Automatically change button type based on available menu choices"""
    def callback(attr, old, new):
        if dropdown.label in pluck_label(new):
            dropdown.button_type = "default"
        else:
            dropdown.button_type = "danger"

    def click(value):
        label = find_label(dropdown.menu, value)
        if label in pluck_label(dropdown.menu):
            dropdown.button_type = "default"
        else:
            dropdown.button_type = "danger"

    dropdown.on_change("menu", callback)
    dropdown.on_click(click)
    return callback


def find_label(menu, value):
    for label, _value in menu:
        if value == _value:
            return label


def pluck_label(menu):
    return [l for l, _ in menu]
