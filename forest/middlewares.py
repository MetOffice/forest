"""Middleware utilities"""


def echo(store, action):
    print(action)
    yield action
