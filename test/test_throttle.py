import datetime as dt
import forest


def fake_now(times):
    generator = (t for t in times)
    def wrapper():
        return next(generator)
    return wrapper


def test_throttle_middleware_given_plain_actions():
    times = [
        dt.datetime(2019, 1, 1, 12, 0, 0),
        dt.datetime(2019, 1, 1, 12, 0, 1),
    ]
    store = forest.redux.Store(forest.db.reducer,
            middlewares=[
                forest.Throttle(now=fake_now(times))
            ])
    actions = [
        forest.db.set_value("key", "a"),
        forest.db.set_value("key", "b"),
    ]
    for action in actions:
        store.dispatch(action)
    assert store.state == {"key": "b"}


def test_throttle_middleware_given_throttled_actions():
    times = [
        dt.datetime(2019, 1, 1, 12, 0, 0),
        dt.datetime(2019, 1, 1, 12, 0, 1),
    ]
    store = forest.redux.Store(forest.db.reducer,
            middlewares=[
                forest.Throttle(now=fake_now(times))
            ])
    actions = [
        forest.db.set_value("key", "a"),
        forest.db.set_value("key", "b"),
    ]
    actions = [forest.throttled(a, 1000) for a in actions]
    for action in actions:
        store.dispatch(action)
    assert store.state == {"key": "a"}
