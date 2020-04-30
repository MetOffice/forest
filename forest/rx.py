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
        observable.add_subscriber(self.notify)
        return self

    def map(self, f):
        """Make new stream by applying f to values

        :returns: new stream that emits f(x)
        """
        stream = Stream()
        def callback(x):
            stream.notify(f(x))
        self.add_subscriber(callback)
        return stream

    def distinct(self, comparator=None):
        """Remove repeated items

        :param comparator: f(x, y) that returns True if x == y
        :returns: new stream with duplicate events removed
        """
        stream = Stream()

        def closure():
            y = None
            called = False
            def callback(x):
                nonlocal y, called
                if not called:
                    y = x  # Important: must be before notify() to prevent recursion
                    stream.notify(x)
                    called = True
                    return
                if comparator is None:
                    not_same = x != y
                else:
                    not_same = not comparator(x, y)
                if not_same:
                    y = x  # Important: must be before notify() to prevent recursion
                    stream.notify(x)
            return callback

        self.add_subscriber(closure())
        return stream

    def filter(self, f):
        """Emit items that pass a predicate test

        :param f: predicate function True keeps value False discards value
        """
        stream = Stream()
        def callback(x):
            if f(x):
                stream.notify(x)
        self.add_subscriber(callback)
        return stream

    @classmethod
    def combine_latest(cls, *input_streams):
        output = cls()
        payload = len(input_streams) * [None]

        def callback(i):
            def wrapper(x):
                nonlocal payload
                payload[i] = x
                output.notify(tuple(payload))
            return wrapper

        for i, stream in enumerate(input_streams):
            stream.add_subscriber(callback(i))

        return output
