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
        data,
        db,
        earth_networks,
        gridded_forecast,
        unified_model,
        rdt,
        satellite)


__all__ = []


@export
class Loader(object):
    """Encapsulates complex Loader construction logic"""
    @classmethod
    def group_args(cls, group, args, database=None):
        """Construct builder from FileGroup and argparse.Namespace

        Simplifies construction of Loaders given command line
        and configuration settings

        :param group: FileGroup instance
        :param args: argparse.Namespace instance
        """
        if group.locator == "database":
            return cls.from_database(
                    database.connection,
                    group.file_type,
                    group.label,
                    group.pattern,
                    replacement_dir=cls.replace_dir(
                        args.directory, group.directory))
        elif group.locator == "file_system":
            if args.config_file is None:
                return cls.from_files(
                        group.label,
                        group.pattern,
                        args.files,
                        group.file_type)
            else:
                pattern = os.path.expanduser(
                        cls.full_pattern(
                            group.pattern,
                            group.directory,
                            args.directory))
                return cls.from_pattern(
                        group.label,
                        pattern,
                        group.file_type)
        else:
            raise Exception("Unknown locator: {}".format(group.locator))

    @classmethod
    def from_database(cls,
            connection,
            file_type,
            label,
            pattern,
            replacement_dir=None):
        """Builds a loader powered by a SQL database

        .. note:: ``replacement_dir`` can be used to modify
                  names in ``file`` table

        :param connection: sqlite3.connection to a database
        :param file_type: keyword to specify particular loader
        :param label: keyword to link app state to loader
        :param replacement_dir: directory to substitute in ``file`` table
        """
        locator = db.Locator(
            connection,
            directory=replacement_dir)
        return cls.file_loader(
                    file_type,
                    pattern,
                    label=label,
                    locator=locator)

    @classmethod
    def from_files(cls, label, pattern, files, file_type):
        """Builds a loader from list of files and a file type"""
        locator = None  # RDT, EIDA50 etc. have built-in locators
        if file_type == 'unified_model':
            locator = unified_model.Locator(files)
        return cls.file_loader(
                    file_type,
                    pattern,
                    label=label,
                    locator=locator)

    @classmethod
    def from_pattern(cls,
            label,
            pattern,
            file_type):
        """Builds a loader from a pattern and a file type"""
        locator = None  # RDT, EIDA50 etc. have built-in locators
        if file_type == 'unified_model':
            locator = unified_model.Locator.pattern(pattern)
        return cls.file_loader(
                    file_type,
                    pattern,
                    label=label,
                    locator=locator)

    @staticmethod
    def file_loader(file_type, pattern, label=None, locator=None):
        file_type = file_type.lower().replace("_", "")
        if file_type == 'rdt':
            return rdt.Loader(pattern)
        elif file_type == 'gpm':
            return data.GPM(pattern)
        elif file_type == 'earthnetworks':
            return earth_networks.Loader.pattern(pattern)
        elif file_type == 'eida50':
            return satellite.EIDA50(pattern)
        elif file_type == 'griddedforecast':
            return gridded_forecast.ImageLoader(label, pattern)
        elif file_type == 'unifiedmodel':
            return data.DBLoader(label, pattern, locator)
        else:
            raise Exception("unrecognised file_type: {}".format(file_type))

    @staticmethod
    def full_pattern(pattern, leaf_dir, prefix_dir):
        """Combine user specified patterns to files on disk

        .. note:: absolute path leaf directory takes precedence over prefix
                  directory

        :param pattern: str representing file name wildcard pattern
        :param leaf_dir: leaf directory to add after prefix directory
        :param prefix_dir: directory to place before leaf and pattern
        """
        dirs = [d for d in [prefix_dir, leaf_dir] if d is not None]
        return os.path.join(*dirs, pattern)

    @staticmethod
    def replace_dir(prefix_dir, leaf_dir):
        """Replacement directory for SQL queries

        Combine two user defined directories to allow flexible
        approach to directory specification

        :param prefix_dir: directory to put before relative leaf directory
        :param leaf_dir: directory to append to prefix
        """
        dirs = [d for d in [prefix_dir, leaf_dir] if d is not None]
        if len(dirs) == 0:
            return
        return os.path.join(*dirs)
