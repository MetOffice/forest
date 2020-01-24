"""Middleware utilities"""


def echo(store, action):
    items = {
        item: store.state.get(item) for item in [
            "variable",
            "initial_time",
            "initial_times",
            "valid_time",
            "valid_times",
            "pressure",
            "pressures"]}
    print(action, items)
    yield action
