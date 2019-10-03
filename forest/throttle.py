import datetime as dt
from forest.redux import middleware
from forest.export import export


__all__ = []


@export
def throttled(action, miliseconds):
    """Add meta-data to action"""
    action["meta"] = {
        "throttle": miliseconds
    }
    return action


@export
class Throttle(object):
    """Middleware to throttle actions"""
    def __init__(self, now=None):
        self.last_seen = {}
        if now is None:
            self.now = dt.datetime.now
        else:
            self.now = now

    @middleware
    def __call__(self, store, next_dispatch, action):
        if "meta" not in action:
            return next_dispatch(action)
        if "throttle" not in action["meta"]:
            return next_dispatch(action)
        current_time = self.now()
        kind = action["kind"]
        if kind not in self.last_seen:
            self.last_seen[kind] = current_time
            return next_dispatch(action)
        else:
            delta = current_time - self.last_seen[action["kind"]]
            elapsed_ms = 1000 * (delta).total_seconds()
            throttle_ms = action["meta"]["throttle"]
            if elapsed_ms > throttle_ms:
                self.last_seen[kind] = current_time
                return next_dispatch(action)
