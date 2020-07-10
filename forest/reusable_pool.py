from forest.object_pool import ObjectPool


class ReusablePool(ObjectPool):
    """Similar to ObjectPool with calls to prepare/reset objects"""
    def acquire(self):
        obj = super().acquire()
        obj.prepare()
        return obj

    def release(self, obj):
        obj.reset()
        super().release(obj)
