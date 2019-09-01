

class Observable(object):
    def __init__(self):
        self.subscribers = []

    def subscribe(self, method):
        self.subscribers.append(method)

    def notify(self, action):
        for method in self.subscribers:
            method(action)
