"""Functional reactive programming"""
from forest.observe import Observable
from forest.export import export


__all__ = []


@export
class Stream(Observable):
    def listen_to(self, observable):
        observable.subscribe(self.notify)
        return self

    def map(self, f):
        stream = Stream()
        def callback(x):
            stream.notify(f(x))
        self.subscribe(callback)
        return stream

    def distinct(self):
        stream = Stream()

        def closure():
            y = None
            def callback(x):
                nonlocal y
                if (y is None) or (x != y):
                    stream.notify(x)
                    y = x
            return callback

        self.subscribe(closure())
        return stream
