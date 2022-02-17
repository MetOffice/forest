"""Middleware utilities"""


def echo(store, action):
    print(action)
    yield action


def echo_state(store, action):
    print(store.state)
    yield action
