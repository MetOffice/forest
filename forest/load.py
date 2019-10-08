"""Factory methods for building Loader classes
"""
import os
from forest.export import export
from forest import (
        data,
        db,
        unified_model)


__all__ = []


@export
class Loader(object):
    """Encapsulates complex Loader construction logic"""
    @classmethod
    def group_args(cls, group, args, database=None):
        """Construct builder from FileGroup and argparse.Namespace

        :param group: FileGroup instance
        :param args: argparse.Namespace instance
        """
        if group.locator == "database":
            return cls.from_database(
                    database,
                    group.file_type,
                    group.label,
                    group.pattern,
                    prefix_dir=args.directory,
                    leaf_dir=group.directory)
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
            database,
            file_type,
            label,
            pattern,
            prefix_dir=None,
            leaf_dir=None):
        locator = db.Locator(
            database.connection,
            directory=cls.replace_dir(prefix_dir, leaf_dir))
        return data.file_loader(
                    file_type,
                    pattern,
                    label=label,
                    locator=locator)

    @classmethod
    def from_files(cls, label, pattern, files, file_type):
        locator = None  # RDT, EIDA50 etc. have built-in locators
        if file_type == 'unified_model':
            locator = unified_model.Locator(files)
        return data.file_loader(
                    file_type,
                    pattern,
                    label=label,
                    locator=locator)

    @classmethod
    def from_pattern(cls,
            label,
            pattern,
            file_type):
        # Search using prefix, leaf and pattern
        locator = None  # RDT, EIDA50 etc. have built-in locators
        if file_type == 'unified_model':
            locator = unified_model.Locator.pattern(pattern)
        return data.file_loader(
                    file_type,
                    pattern,
                    label=label,
                    locator=locator)

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
