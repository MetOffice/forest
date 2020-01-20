"""
Observer pattern
----------------

The most basic observer pattern simply calls
one at a time each subscriber with a value
that represents information important to the
subscriber.

.. autoclass:: Observable
   :members:

"""

class Observable(object):
    """Basic observer design pattern"""
    def __init__(self):
        self.subscribers = []

    def add_subscriber(self, method):
        """Append method to list of subscribers"""
        self.subscribers.append(method)

    def notify(self, value):
        """Call subscribers with value"""
        for method in self.subscribers:
            method(value)
