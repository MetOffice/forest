"""
Loader factory
--------------

To make it simpler to construct a Loader, a factory class
has been written with ``@classmethods`` designed to build
loaders appropriate for each supported visualisation/file type

>>> loader = forest.Loader.from_pattern("Label", "*.json", "rdt")
>>> isinstance(loader, forest.rdt.Loader)
... True

Abstracting the construction of various Loader classes away
from ``main.py`` allows re-usability and finer grained
testing.


.. autoclass:: Loader
   :members:

"""
import os
from forest.export import export
from forest import (
        exceptions,
        data,
        gridded_forecast,
        rdt,
        intake_loader,
        nearcast)


__all__ = []


@export
class Loader(object):
    """Encapsulates complex Loader construction logic"""
    @classmethod
    def group_args(cls, group, args):
        """Construct builder from FileGroup and argparse.Namespace

        Simplifies construction of Loaders given command line
        and configuration settings

        :param group: FileGroup instance
        :param args: argparse.Namespace instance
        """
        if args.config_file is None:
            return cls.from_files(
                    group.label,
                    group.pattern,
                    args.files,
                    group.file_type)
        else:
            pattern = os.path.expanduser(group.pattern)
            return cls.from_pattern(
                    group.label,
                    pattern,
                    group.file_type)

    @classmethod
    def from_files(cls, label, pattern, files, file_type):
        """Builds a loader from list of files and a file type"""
        return cls.file_loader(
                    file_type,
                    pattern,
                    label=label)

    @classmethod
    def from_pattern(cls,
            label,
            pattern,
            file_type):
        """Builds a loader from a pattern and a file type"""
        return cls.file_loader(
                    file_type,
                    pattern,
                    label=label)

    @staticmethod
    def file_loader(file_type, pattern, label=None):
        file_type = file_type.lower().replace("_", "")
        if file_type == 'rdt':
            return rdt.Loader(pattern)
        elif file_type == 'gpm':
            return data.GPM(pattern)
        elif file_type == 'griddedforecast':
            return gridded_forecast.ImageLoader(label, pattern)
        elif file_type == 'intake':
            return intake_loader.IntakeLoader(pattern)
        elif file_type == 'nearcast':
            return nearcast.NearCast(pattern)
        else:
            raise exceptions.UnknownFileType("unrecognised file_type: {}".format(file_type))
