

class Observable(object):
    def __init__(self):
        self.subscribers = []

    def subscribe(self, method):
        self.subscribers.append(method)

    def notify(self, value):
        for method in self.subscribers:
            method(value)
