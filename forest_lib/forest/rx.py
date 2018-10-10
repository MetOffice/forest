"""Functional reactive programming utilities"""
__all__ = [
    "Stream"
]


class Stream(object):
    def __init__(self):
        self.subscribers = []

    def register(self, subscriber):
        self.subscribers.append(subscriber)

    def emit(self, value):
        for subscriber in self.subscribers:
            subscriber.notify(value)

    def log(self):
        return Log(self)

    def scan(self, initial, combinator):
        return Scan(self, initial, combinator)

    def map(self, transform):
        return Map(self, transform)

    def filter(self, criteria):
        return Filter(self, criteria)

    def unique(self):
        return Unique(self)


class Unique(Stream):
    def __init__(self, stream):
        stream.register(self)
        super().__init__()

    def notify(self, value):
        if (not hasattr(self, 'last')) or (self.last != value):
            self.last = value
            self.emit(value)


class Log(Stream):
    def __init__(self, stream):
        stream.register(self)
        super().__init__()

    def notify(self, value):
        print(value)
        self.emit(value)


class Map(Stream):
    def __init__(self, stream, transform):
        self.transform = transform
        stream.register(self)
        super().__init__()

    def notify(self, value):
        self.emit(self.transform(value))


class Filter(Stream):
    def __init__(self, stream, criteria):
        self.criteria = criteria
        stream.register(self)
        super().__init__()

    def notify(self, value):
        if not self.criteria(value):
            self.emit(value)


class Scan(Stream):
    def __init__(self, stream, initial, combinator):
        self.state = initial
        self.combinator = combinator
        stream.register(self)
        super().__init__()

    def notify(self, value):
        self.state = self.combinator(self.state, value)
        self.emit(self.state)
