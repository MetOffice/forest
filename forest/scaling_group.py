"""Scale a group of active objects up/down as required"""


class ScalingGroup:
    """Scale collection of objects up/down to meet criteria"""
    def __init__(self, object_pool):
        self.object_pool = object_pool
        self.instances = []

    def scale_to(self, number):
        """Adjust number of instances to match number"""
        current_number = len(self.instances)
        if current_number < number:
            # Scale up
            for _ in range(number - current_number):
                obj = self.object_pool.acquire()
                self.instances.append(obj)

        elif current_number > number:
            # Scale down
            for obj in self.instances[number:]:
                self.object_pool.release(obj)
            self.instances = self.instances[:number]
