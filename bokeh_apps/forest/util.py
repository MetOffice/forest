from functools import partial


class Observable(object):
    def __init__(self):
        self.uid = 0
        self.listeners = []

    def subscribe(self, listener):
        self.uid += 1
        self.listeners.append(listener)
        return partial(self.unsubscribe, int(self.uid))

    def unsubscribe(self, uid):
        del self.listeners[uid]

    def announce(self, *args):
        for listener in self.listeners:
            listener(*args)


def select(dropdown):
    def wrapped(new):
        for label, value in dropdown.menu:
            if value == new:
                dropdown.label = label
    return wrapped
