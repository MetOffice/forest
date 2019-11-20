"""Functional reactive programming"""
from forest.observe import Observable


class Stream(Observable):
    def listen_to(self, observable):
        observable.subscribe(self.notify)
        return self

    def map(self, f):
        """Make new stream by applying f to values"""
        stream = Stream()
        def callback(x):
            stream.notify(f(x))
        self.subscribe(callback)
        return stream

    def distinct(self):
        """Remove repeated items"""
        stream = Stream()

        def closure():
            y = None
            called = False
            def callback(x):
                nonlocal y, called
                if (not called) or (x != y):
                    stream.notify(x)
                    y = x
                    called = True
            return callback

        self.subscribe(closure())
        return stream

    def filter(self, f):
        """Emit items that pass a predicate test"""
        stream = Stream()
        def callback(x):
            if f(x):
                stream.notify(x)
        self.subscribe(callback)
        return stream
