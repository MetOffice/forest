"""Middleware utilities"""


class Log:
    def __init__(self):
        self.actions = []

    def __call__(self, store, action):
        self.actions.append(action)
        yield action


def echo(store, action):
    print(action)
    yield action
