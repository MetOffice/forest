from abc import ABC, abstractmethod


class Reusable(ABC):
    """Re-usable objects are prepared when acquired and reset when released"""
    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def reset(self):
        pass
