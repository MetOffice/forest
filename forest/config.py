"""Application configuration"""
import yaml


__all__ = []


def export(obj):
    if obj.__name__ not in __all__:
        __all__.append(obj.__name__)
    return obj


class Config(object):
    def __init__(self, data):
        self.data = data

    @property
    def patterns(self):
        if "files" in self.data:
            return [(f["name"], f["pattern"])
                    for f in self.data["files"]]
        return []

    @classmethod
    def load(cls, path):
        with open(path) as stream:
            data = yaml.load(stream)
        return cls(data)

@export
def load_config(path):
    return Config.load(path)
