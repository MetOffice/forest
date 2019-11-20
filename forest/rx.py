"""
Rx - Functional reactive programming
------------------------------------

The basic data structue in functional reactive
programming is a :class:`Stream`

.. autoclass:: Stream
    :members:

"""
from forest.observe import Observable


class Stream(Observable):
    """Sequence of events

    An event is a value passed to a callback or listener. Most
    data structures exist in space, e.g. in RAM or on disk, but
    Streams exist in time.

    Common operations on streams include :func:`~Stream.map` and :func:`~Stream.filter`.
    """
    def listen_to(self, observable):
        """Re-transmit events on another observable

        :returns: current stream
        """
        observable.subscribe(self.notify)
        return self

    def map(self, f):
        """Make new stream by applying f to values

        :returns: new stream that emits f(x)
        """
        stream = Stream()
        def callback(x):
            stream.notify(f(x))
        self.subscribe(callback)
        return stream

    def distinct(self):
        """Remove repeated items

        :returns: new stream with duplicate events removed
        """
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
        """Emit items that pass a predicate test

        :param f: predicate function True keeps value False discards value
        """
        stream = Stream()
        def callback(x):
            if f(x):
                stream.notify(x)
        self.subscribe(callback)
        return stream
