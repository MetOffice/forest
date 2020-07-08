"""Object Pool design pattern"""


class ObjectPool:
    """Object Pool design pattern"""
    def __init__(self, factory):
        self.factory = factory
        self._objects = []

    def acquire(self):
        """Construct or re-use an object"""
        if len(self._objects) == 0:
            return self.factory()
        else:
            return self._objects.pop()

    def release(self, obj):
        """Place object back in the pool"""
        self._objects.append(obj)
