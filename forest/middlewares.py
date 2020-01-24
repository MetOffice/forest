"""Middleware utilities"""


def echo(store, action):
    items = {
        item: store.state.get(item) for item in [
            "variable",
            "initial_time",
            "valid_time",
            "pressure",
            "pressures"]}
    print(action, items)
    yield action
