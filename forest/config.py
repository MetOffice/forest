"""Application configuration"""
import os
import yaml


__all__ = []


def export(obj):
    if obj.__name__ not in __all__:
        __all__.append(obj.__name__)
    return obj


class Config(object):
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return "{}({})".format(
                self.__class__.__name__,
                self.data)

    @property
    def patterns(self):
        if "files" in self.data:
            return [(f["label"], f["pattern"])
                    for f in self.data["files"]]
        return []

    @classmethod
    def load(cls, path):
        with open(path) as stream:
            data = yaml.load(stream)
        return cls(data)

    @property
    def file_groups(self):
        return [FileGroup(**data)
                for data in self.data["files"]]


class FileGroup(object):
    def __init__(self,
            label,
            pattern,
            locator="file_system",
            file_type="unified_model",
            directory=None):
        self.label = label
        self.pattern = pattern
        self.locator = locator
        self.file_type = file_type
        self.directory = directory

    @property
    def full_pattern(self):
        if self.directory is None:
            return self.pattern
        return os.path.join(self.directory, self.pattern)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise Exception("Can not compare")
        attrs = ("label", "pattern", "locator", "file_type", "directory")
        return all(
                getattr(self, attr) == getattr(other, attr)
                for attr in attrs)

    def __repr__(self):
        arg_attrs = [
            "label",
            "pattern"]
        args = [self._str(getattr(self, attr))
                for attr in arg_attrs]
        kwarg_attrs = [
            "locator",
            "file_type",
            "directory"]
        kwargs = [
            "{}={}".format(attr, self._str(getattr(self, attr)))
                for attr in kwarg_attrs]
        return "{}({})".format(
                self.__class__.__name__,
                ", ".join(args + kwargs))
    @staticmethod
    def _str(value):
        if isinstance(value, str):
            return "'{}'".format(value)
        else:
            return str(value)


@export
def load_config(path):
    return Config.load(path)
